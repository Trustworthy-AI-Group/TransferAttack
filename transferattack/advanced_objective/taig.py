import torch

from ..utils import *
from ..attack import Attack

class TAIG(Attack):
    """
    TAIG Attack
    'Transferable Adversarial Attack based on Integrated Gradients'(https://arxiv.org/abs/2205.13152)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        steps (int): the number of simulations of integral
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, num_scale=20, epoch=10, decay=1, steps=20.
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, steps=20,**kwargs):
        super().__init__('TAIG', model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 0
        self.steps = steps

    def compute_ig(self, data, delta,label_inputs):
        baseline = torch.zeros_like(data)
        scaled_inputs = [baseline + (float(i) / self.steps) * (data+delta - baseline).detach() for i in
                         range(0, self.steps + 1)]
        scaled_inputs = torch.stack(scaled_inputs,dim=0)
        scaled_inputs.requires_grad = True
        att_out = self.model(scaled_inputs)
        score = att_out[:, label_inputs]
        loss = -torch.mean(score)
        self.model.zero_grad()
        loss.backward()
        grads = scaled_inputs.grad.data
        avg_grads = torch.mean(grads, dim=0)
        delta_X = scaled_inputs[-1] - scaled_inputs[0]
        integrated_grad = delta_X * avg_grads
        IG = integrated_grad.detach()
        del integrated_grad,delta_X,avg_grads,grads,loss,score,att_out
        return IG
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        momentum = 0.
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.epoch):
            grad = []
            for x,d,y in zip(data,delta,label):
                grad.append(self.compute_ig(x,d,y))
            grad = torch.stack(grad,dim=0)
            self.model.zero_grad()
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        # exit()
        return delta.detach()