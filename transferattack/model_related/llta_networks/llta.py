import torch
import torch.nn.functional as F
import random

from ...utils import *
from ...gradient.mifgsm import MIFGSM

from .models import decayresnet, decaydensenet, Gaussian, get_Gaussian_kernel, Gamma_Wrap_for_ResNet, Gamma_Wrap_for_DenseNet

gamma_num_dic = {
    "resnet18": 4,  
    "resnet34": 12, 
    "resnet50": 12, 
    "resnet101": 29,
    "resnet152": 46,

    "densenet121": 54,
    "densenet169": 78,
    "densenet201": 94,
}
GMIN = 0.0
GMAX = 1e5


class LLTA(MIFGSM):
    """
    LLTA Attack
    'Learning to Learn Transferable Attack (AAAI 2022)'(https://cdn.aaai.org/ojs/19936/19936-13-23949-1-2-20220628.pdf)

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
        device (torch.device): the device for data. If it is None, the device would be same as model
        attack (str): the name of attack.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.,

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/llta/resnet50 --attack llta --model=resnet50 --batchsize 1
        python main.py --input_dir ./path/to/data --output_dir adv_data/llta/resnet50 --eval
    Notes:
        The batchsize should be set to 1!
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='LLTA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.gamma_num = gamma_num_dic[model_name]
        self.inner_iters = 5
        self.nsample = 5
        self.spt_size = 20
        self.spt_prob_m = 0.5
        self.spt_region = 0.1
        self.spt_prob_d = 0.5
        self.qry_size = 10
        self.qry_prob_m = 0.5
        self.qry_region = 0.1
        self.qry_prob_d = 0.5
        self.task_num = 5
        self.sigma = 0.05

    def load_model(self, model_name):
        if 'resnet' in model_name:
            model = getattr(decayresnet, model_name)(pretrained=True)
            return Gamma_Wrap_for_ResNet(model_name, model.eval().cuda())
        elif 'densenet':
            model = getattr(decaydensenet, model_name)(pretrained=True)
            return Gamma_Wrap_for_DenseNet(model_name, model.eval().cuda())
        else:
            raise ValueError("The model_name should be resnet or densenet!")

    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

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

        batch_size, c, w, h = data.shape

        momentum = 0
        global gaussian 
        gaussian = Gaussian(loc=0., scale=self.sigma)
        for _ in range(self.epoch):
            gammas = torch.full((batch_size, self.gamma_num), 0.5, dtype=torch.float32, device=self.device) # reinit

            for epc in range(self.inner_iters):
                gamma_delta = self.optimize_parameter(data + delta, label, self.model, gammas, nsample=self.nsample)
                gammas = torch.clamp(gammas + gamma_delta, GMIN, GMAX)

            # create suppory set
            spt_gammas = self.create_model_task_set(
                gammas,
                set_size=self.spt_size,
                prob=self.spt_prob_m,
                region=self.spt_region,
            )
            spt_data = self.create_data_task_set(
                (data + delta).data,
                set_size=self.spt_size,
                prob=self.spt_prob_d,
            )

            # create query set
            qry_gammas = self.create_model_task_set(
                gammas, 
                set_size=self.qry_size,
                prob=self.qry_prob_m,
                region=self.qry_region,
            )
            qry_data = self.create_data_task_set(
                (data + delta).data,
                set_size=self.qry_size,
                prob=self.qry_prob_d,
            )

            y_repeat = label.unsqueeze(0).repeat(self.qry_size, 1)
            y_expand = torch.cat([
                y_repeat[:, ii]
                for ii in range(batch_size)
            ])

            grads = torch.zeros_like(data)

            for _ in range(self.task_num):
                # sample batch of tasks
                select_idx = torch.as_tensor([
                    random.sample(range(bs*self.spt_size, (bs+1)*self.spt_size), self.qry_size)
                    for bs in range(batch_size)
                ])
                select_idx = select_idx.view(-1)
                spt_batch_gammas = spt_gammas[select_idx]
                spt_batch_x = spt_data[select_idx]

                # get gradient on batch support set
                spt_delta = torch.zeros_like(spt_batch_x)
                spt_delta.requires_grad_()  
                spt_loss = F.cross_entropy(self.model(spt_batch_x.data + spt_delta, spt_batch_gammas), y_expand, reduction='sum')
                spt_loss.backward()
                spt_delta.data = spt_delta.data + self.epsilon * spt_delta.grad.sign()
                spt_delta.data = torch.clamp(spt_delta.data, -self.epsilon, self.epsilon)
                spt_delta.data = torch.clamp(spt_batch_x.data + spt_delta, 0., 1.) - spt_batch_x.data 
                # import ipdb; ipdb.set_trace()

                # accumulate loss on query set
                qry_loss = F.cross_entropy(self.model(qry_data.data + spt_delta, qry_gammas), y_expand, reduction='sum')
                qry_loss.backward()

                grads += spt_delta.grad.data.view(self.qry_size, batch_size, c, w, h).sum(0)
                spt_delta.grad.data.zero_()

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    
    def optimize_parameter(self, x, y, model, gammas, nsample):
        # to ensure x and y are leaf nodes
        x = x.data
        y = y.data
        device = x.device

        cur_l2grad = self.get_l2grad(x, y, self.model, gammas)

        gamma_delta = gaussian.sample((nsample, x.shape[0], gammas.shape[-1]), device=device)
        prob_q = gaussian.prob(gamma_delta)
        new_l2grad = torch.stack([
            self.get_l2grad(x, y, self.model, gammas=torch.clamp(gammas + gd, GMIN, GMAX))
            for gd in gamma_delta
        ])

        diff = new_l2grad - cur_l2grad
        prob_p = ((-diff / 1.0).exp() * (diff < 0)).unsqueeze(-1)

        opt_gamma_delta = ((prob_p / prob_q) * gamma_delta).sum(0)
        Z = (prob_p / prob_q).sum(0)
        return opt_gamma_delta / (Z + 1e-12)
    
    def create_model_task_set(self, gammas, set_size, prob, region):
        device = gammas.device
        aug_size = int(set_size*prob)
        gamma_sets = []
        for gamma in gammas:
            gamma_set = gamma.repeat(set_size, 1)
            aug_delta = (torch.rand_like(gamma_set) - 0.5) * 2 * region
            aug_mask = (torch.rand((set_size, 1), device=device) < prob)
            gamma_sets.append(gamma_set + aug_mask*aug_delta)
        # import ipdb; ipdb.set_trace()
        gamma_sets = torch.cat(gamma_sets)
        gamma_sets = torch.clamp(gamma_sets, GMIN, GMAX)
        return gamma_sets

    def create_data_task_set(self, x, set_size, prob):
        def input_diversity(img):
            size = img.size(2)
            resize = int(size / 0.875)

            rnd = torch.randint(size, resize + 1, (1,)).item()
            rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
            h_rem = resize - rnd
            w_hem = resize - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_hem + 1, (1,)).item()
            pad_right = w_hem - pad_left
            padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
            padded = F.interpolate(padded, (size, size), mode="nearest")
            return padded

        aug_ls = []
        for _ in range(set_size):
            p = torch.rand(1).item()
            if p >= prob:
                aug_ls.append(x)
            else:
                aug_ls.append(input_diversity(x))
        aug_ls = torch.stack(aug_ls)
        inputs = torch.cat([
            aug_ls[:, i, :, :, :]
            for i in range(aug_ls.shape[1])
        ])
        return inputs

    def get_l2grad(self, x, y, model, gammas):
        zero_delta = torch.zeros_like(x)
        zero_delta.requires_grad_()
        loss = F.cross_entropy(model(x+zero_delta, gammas), y, reduction='sum')
        loss.backward()
        grad = zero_delta.grad.data
        l2grad = grad.norm(p=2, dim=(1,2,3)) 
        return l2grad
