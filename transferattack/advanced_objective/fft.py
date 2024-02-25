import torch

from ..utils import *
from ..attack import Attack
import torch.nn.functional as F
import scipy.stats as st

from torch.nn.modules.module import Module

mid_outputs = None

class FFT(Attack):
    """
    FFT (Feature space fine-tuning)
    'Enhancing Targeted Transferability via Feature Space Fine-tuning (ICASSP 2024)'(https://arxiv.org/abs/2401.02727)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        coeff (float): coefficient.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        beta_combine: the constant to combine the AG of x_adv with the AG of x_ori.
        epoch_ft: fine-tuning epochs
        alpha_ft: fine-tuning step size

    Official arguments:
        epsilon=16/255, alpha=2.0/255, epoch=300(baseline attack) , decay=1., coeff=1.0
        beta_combine=0.2, epoch_ft=10, alpha_ft = alpha/2

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/fft/resnet18_targeted --attack fft --model=resnet18 --targeted
        python main.py --input_dir ./path/to/data --output_dir adv_data/fft/resnet18_targeted --eval --targeted

    NOTE:
        1). FFT is only useful for TARGETED attack. It does not make sense to try to boost untargeted attack with it.
        2). About the middle layer to be attacked: the principle is the middle-to-end module. For a model with 4 blocks, the 3rd block is a good candidate.
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=2.0 / 255, random=False, epoch=300, decay=1., coeff=1.0, 
                 drop_rate=0.3, num_ens=30, beta_combine=0.2, epoch_ft=10, targeted=False, random_start=False, norm='linfty', 
                 loss='crossentropy', device=None, attack='FFT', loss_base='logit_margin', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.random = random
        self.coeff = coeff
        self.model_name = model_name

        self.num_ens = num_ens  # ensemble number for AG, following FIA
        self.drop_rate = drop_rate  # 0.3, following FIA
        self.beta_combine = beta_combine  # following SupHigh method
        self.alpha_ft = alpha / 2  # should < self.alpha
        self.epoch_ft = epoch_ft  # should << self.epoch
        self.loss_base = loss_base  # the loss function of the baseline attack, 202402

    # define DI
    def DI_keepresolution(self, X_in):
        img_size = X_in.shape[-1]
        rnd = np.random.randint(img_size - 22, img_size)
        h_rem = img_size - rnd
        w_rem = img_size - rnd
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c <= 0.7:
            X_in_inter = F.interpolate(X_in, size=(rnd, rnd))
            X_out = F.pad(X_in_inter, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            return X_out
        else:
            return X_in

    # define TI
    def gkern(self, kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    # obtain gaussian_kernel
    def get_Gaussian_kernel(self, kernel_size=5):
        # kernel_size = 5
        kernel = self.gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        return gaussian_kernel

    # redefine the transform function
    def transform(self, data, **kwargs):
        return self.DI_keepresolution(data)

    # redefine the get_grad function
    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        gaussian_kernel = self.get_Gaussian_kernel(kernel_size=5)
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
        return grad

    ###### FIA related function
    def __backward_hook(self, m, i, o):
        global mid_grad
        mid_grad = o

    def drop(self, data):
        x_drop = torch.zeros(data.size()).cuda()
        x_drop.copy_(data).detach()
        x_drop.requires_grad = True
        Mask = torch.bernoulli(torch.ones_like(x_drop) * (1 - self.drop_rate))
        x_drop = x_drop * Mask
        return x_drop

    ####### FIA related function end

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        # baseline attack. Literally, the power of fine-tuned AE is more depended on the baseline attack, rather
        # than the fine-tuning scheme.
        # How to set the loss function of base attack?
        # attacker = transferattack.attack_zoo[args.attack.lower()](..., loss_base='logit_margin')

        # 202402 modified
        if self.loss_base == 'CE':
            init_delta = super().forward(data, label, **kwargs)
        elif self.loss_base == 'logit':   # default
            self.get_loss = LogitLoss()
            init_delta = super().forward(data, label, **kwargs)
        elif self.loss_base == 'logit_margin':
            self.get_loss = Logit_marginLoss()
            init_delta = super().forward(data, label, **kwargs)
        else:
            raise ValueError("Only support three types of loss functions now: CE, logit, logit_margin.")

        if self.targeted:
            assert len(label) == 2
            label_ori = label[0]
            label_tar = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label_ori = label_ori.clone().detach().to(self.device)
        label_tar = label_tar.clone().detach().to(self.device)

        # 1 Initialize adversarial perturbation
        delta = self.init_delta(data)

        ### 2.1 Aggregate Gradient of X_ori
        if self.model_name in ['resnet18', 'resnet50']:
            """res18, 50, the output of the 3rd Block (total 4)"""
            h2 = self.model[1]._modules.get('layer3')[-1].register_full_backward_hook(self.__backward_hook)
        elif self.model_name in ['densenet121']:
            """dense 121 the output of the 3rd denseBlock (total 4)"""
            h2 = self.model[1]._modules.get('features')[7].register_full_backward_hook(self.__backward_hook)
        elif self.model_name in ['inception_v3']:
            """incV3  'Mixed_6b'"""
            h2 = self.model[1]._modules.get('Mixed_6b').register_full_backward_hook(self.__backward_hook)
        elif self.model_name in ['vgg16_bn']:
            """vgg16_bn, the end of the 4th block (total 5)"""
            h2 = self.model[1]._modules.get('features')[33].register_full_backward_hook(self.__backward_hook)
        else:
            raise ValueError("Please select the correct model! (e.g., resnet18, resnet50, densenet121, etc.)")

        agg_grad = 0
        for _ in range(self.num_ens):
            # 202402 modified
            img_temp_i = self.model[0](data).clone()
            x_drop = self.drop(img_temp_i)
            output_random = self.model[1](x_drop)
            # get the logit of the corresponding label
            logit_label = output_random.gather(1, label_ori.unsqueeze(1)).squeeze(1)
            loss = logit_label.sum()

            self.model.zero_grad()
            loss.backward()
            agg_grad += mid_grad[0].detach()

        for batch_i in range(data.shape[0]):
            agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
        h2.remove()

        ### 2.2 Aggregate Gradient of X_adv
        if self.model_name in ['resnet18', 'resnet50']:
            """res18, 50, the output of the 3rd Block (total 4)"""
            h2 = self.model[1]._modules.get('layer3')[-1].register_full_backward_hook(self.__backward_hook)
        elif self.model_name in ['densenet121']:
            """dense 121 the output of the 3rd denseBlock (total 4)"""
            h2 = self.model[1]._modules.get('features')[7].register_full_backward_hook(self.__backward_hook)
        elif self.model_name in ['inception_v3']:
            """incV3  'Mixed_6b'"""
            h2 = self.model[1]._modules.get('Mixed_6b').register_full_backward_hook(self.__backward_hook)
        elif self.model_name in ['vgg16_bn']:
            """vgg16_bn, the end of the 4th block (total 5)"""
            h2 = self.model[1]._modules.get('features')[33].register_full_backward_hook(self.__backward_hook)
        else:
            raise ValueError("Please select the correct model! (e.g., resnet18, resnet50, densenet121, etc.)")

        agg_grad_adv = 0
        for _ in range(self.num_ens):
            # 202402 modified
            img_temp_i = self.model[0](data + init_delta).clone()
            x_drop = self.drop(img_temp_i)
            output_random = self.model[1](x_drop)
            # get the logit of the corresponding label
            logit_label = output_random.gather(1, label_tar.unsqueeze(1)).squeeze(1)
            loss = logit_label.sum()

            self.model.zero_grad()
            loss.backward()
            agg_grad_adv += mid_grad[0].detach()
        for batch_i in range(data.shape[0]):
            agg_grad_adv[batch_i] /= agg_grad_adv[batch_i].norm(2)
        h2.remove()

        ### 2.3 Combined AG, self.beta_combine should be smaller than 1.
        agg_grad_combine = agg_grad_adv - self.beta_combine * agg_grad
        ### End of AG

        global mid_outputs
        hs = []

        def get_mid_output(model_, input_, o):
            global mid_outputs
            mid_outputs = o

        ## 3 Fine-tune begin
        if self.model_name in ['resnet18', 'resnet50']:
            hs.append(self.model[1]._modules.get('layer3')[-1].register_forward_hook(get_mid_output))  # resnet
        elif self.model_name in ['densenet121']:
            hs.append(self.model[1]._modules.get('features')[7].register_forward_hook(get_mid_output))  # dense121
        elif self.model_name in ['inception_v3']:
            hs.append(self.model[1]._modules.get('Mixed_6b').register_forward_hook(get_mid_output))  # incV3
        elif self.model_name in ['vgg16_bn']:
            hs.append(self.model[1]._modules.get('features')[33].register_forward_hook(get_mid_output))  # vgg16_bn
        else:
            raise ValueError("Please select the correct model! (e.g., resnet18, resnet50, densenet121, etc.)")

        data_adv = data + init_delta
        momentum = 0
        for _ in range(self.epoch_ft):
            # Obtain the output
            logits = self.get_logits(self.transform(data_adv + delta))  # DI

            # Calculate the loss
            loss = torch.sum(agg_grad_combine * mid_outputs)  # FIA loss

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            # Note, the overall perturbation (not only delta) should be bounded, ZH
            delta = torch.clamp(init_delta + delta + self.alpha_ft * momentum.sign(), -self.epsilon,
                                self.epsilon) - init_delta
            delta = clamp(delta, img_min - data_adv, img_max - data_adv)

            mid_outputs = []

        for h in hs:
            h.remove()
        ## 3 Fine-tune end
        return (init_delta + delta).detach()
        # when you need compare the AE w/ and w/o fine-tuning
        # return (init_delta + delta).detach(), init_delta.detach()


# Advanced, targeted attack-tailored loss functions
class LogitLoss(Module):
    """
    targeted logit loss
    """
    def __init__(self):
        super(LogitLoss, self).__init__()

    def forward(self, logits, label):
        logit_tar = logits.gather(1, label.unsqueeze(1)).squeeze(1)
        loss = logit_tar.sum()

        return loss


class Logit_marginLoss(Module):
    """
    targeted margin-calibrated loss
    """
    def __init__(self):
        super(Logit_marginLoss, self).__init__()

    def forward(self, logits, label):
        value, _ = torch.sort(logits, dim=1, descending=True)
        logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()  # margin-calibrated loss
        loss = torch.nn.CrossEntropyLoss(reduction='sum')(logits, label)

        return -loss
