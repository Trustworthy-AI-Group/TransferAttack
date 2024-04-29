from ..utils import *
from ..attack import Attack

class CWA(Attack):
    """
    CWA Attack
    'Rethinking Model Ensemble in Transfer-based Adversarial Attacks'(https://arxiv.org/abs/2303.09105)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=2*epsilon/epoch=3.2/255, epoch=10, beta=50, r_size=16/255/15, inner_step_size=250

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/cwa/ens --attack cwa --model='resnet18,resnet101,resnext50_32x4d,densenet121'
        python main.py --input_dir ./path/to/data --output_dir adv_data/cwa/ens --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=3.2/255, epoch=10, decay=1.0, beta=50, r_size=16/255/15, inner_step_size=250, targeted=False, random_start=True, 
                 norm='linfty', loss='crossentropy', device=None, attack='CWA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.beta = beta
        self.r_size = r_size
        self.inner_step_size = inner_step_size
        self.K = len(model_name)
        

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        inner_momentum = 0.
        outer_momentum = 0.

        delta = self.init_delta(data).to(self.device)

        for _ in range(self.epoch):
            original_delta = delta.clone().detach().to(self.device)

            """Calculate the gradient of the ensemble model"""
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            inner_delta = delta.clone().detach()
            inner_delta = self.update_delta(inner_delta, data, grad, -self.r_size)

            for k in range(self.K):
                inner_delta.requires_grad = True
                inner_k_logits = self.get_logits_by_model_k(self.transform(data+inner_delta), k)
                inner_k_grad = self.get_grad(self.get_loss(inner_k_logits, label), inner_delta)
                inner_delta.requires_grad = False
                """Update the inner gradient by momentum"""
                # inner_momentum = self.get_momentum(inner_k_grad, inner_momentum)
                inner_momentum = self.decay * inner_momentum + inner_k_grad / torch.norm(inner_k_grad.reshape(data.shape[0], -1), p=2, dim=1).view(data.shape[0], 1, 1, 1)
        
                inner_delta = torch.clamp(inner_delta + self.inner_step_size * inner_momentum, -self.epsilon, self.epsilon)
                inner_delta = clamp(inner_delta, img_min-data, img_max-data)
                # delta.requires_grad = True
            """Calculate the update gradient"""
            fake_grad = inner_delta-original_delta
            outer_momentum = outer_momentum * self.decay + fake_grad / torch.norm(fake_grad, p=1)

            """Update the outer gradient by momentum"""
            delta = self.update_delta(delta, data, outer_momentum, self.alpha)

        return delta.detach()

    def get_logits_by_model_k(self, x, k):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.model.models[k](x)
