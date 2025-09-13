import torch
from ..utils import *
from ..attack import Attack


class AWT(Attack):
    """
    AWT Attack
    'Enhancing Adversarial Transferability with Adversarial Weight Tuning (AAAI 2025)'(https://ojs.aaai.org/index.php/AAAI/article/view/32203/34358)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        gamma (float): the balanced coefficient.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        sam_lr (float): learning rate for SAM inner optimizer.
        sam_rho (float): perturbation radius for SAM.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=3.0, gamma=0.5, num_neighbor=20, epoch=10, decay=1.
        For ResNet50, sam_lr=0.002, sam_rho=0.005. For Inception-v3, sam_lr=0.001, sam_rho=0.002
    
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/awt/resnet50 --attack awt --model resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/awt/resnet50 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=3.0, gamma=0.5, num_neighbor=20, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='AWT', sam_lr=0.002, sam_rho=0.005, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.sam_lr = sam_lr
        self.sam_rho = sam_rho
        self.sam = SAM(self.model.parameters(), torch.optim.SGD, lr = self.sam_lr,rho=self.sam_rho, momentum=0.5)

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        averaged_gradient = 0
        
        for idx in range(self.num_neighbor):
            x_near = self.transform(data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))
            logits = self.get_logits(x_near)
            loss = self.get_loss(logits, label)
            g_1 = self.get_grad(loss, delta)

            x_next = self.transform(x_near + self.alpha*(-g_1 / (torch.abs(g_1).mean(dim=(1,2,3), keepdim=True))))
            logits = self.get_logits(x_next)
            loss = self.get_loss(logits, label)
            g_2 = self.get_grad(loss, delta)
            averaged_gradient += (1-self.gamma)*g_1 + self.gamma*g_2

        return averaged_gradient / self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The AWT attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        self.sam.save_params()
        momentum, averaged_gradient = 0, 0
        for _ in range(self.epoch):
            def closure():
                logits = self.get_logits(self.transform(data+delta, momentum=momentum))
                loss = self.get_loss(logits, label) + self.get_loss(self.get_logits(self.transform(data, momentum=momentum)), label)
                loss.backward(retain_graph=True)
                return loss, logits
            
            # Obtain the output and calculate the loss
            loss, logits = closure()
            self.sam.step(closure=closure)
            # Calculate the averaged updated gradient
            averaged_gradient = self.get_averaged_gradient(data, delta, label)
            # Calculate the momentum
            momentum = self.get_momentum(averaged_gradient, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        self.sam.recover_step()
        
        return delta.detach()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        loss, logits = closure()
        self.second_step()
        return loss, logits

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

        
    def save_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old"] = p.data.clone()

    @torch.no_grad()
    def recover_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old"]

        if zero_grad: self.zero_grad()
