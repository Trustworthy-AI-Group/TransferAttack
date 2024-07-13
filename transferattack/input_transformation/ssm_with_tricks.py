import torch
import torch.nn.functional as F

from ..gradient.mifgsm import MIFGSM
from ..utils import *
"""
Results of the SSM with tricks on this library:
| 100.0 | 69.9 | 73.7 | 93.6 | 29.6 | 40.1 | 55.7 | 62.3 |  SSM
| 100.0 | 72.4 | 75.0 | 94.0 | 32.7 | 43.7 | 61.4 | 65.7 |  SSM with high frequency perturbation (SSM_H)
| 100.0 | 76.8 | 80.6 | 96.0 | 39.7 | 48.2 | 64.7 | 69.6 |  SSM with block perturbation (SSM_P)

Note: The implementation for learnable spectrum simulation can be referred to the l2t.py
"""




class SSM_H(MIFGSM):
    """
    High-frequency SSM Attack
    'Bag of tricks to boost the adversarial transferability'(https://arxiv.org/abs/2401.08734)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_spectrum (int): the number of spectrum.
        rho (float): the tuning factor for Uniform distribution.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, num_spectrum=20, rho=0.5

    Example script:
        python main.py --attack ssm --output_dir adv_data/ssm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_spectrum=20, rho=0.5, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None,  **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device)
        self.num_spectrum = num_spectrum
        self.epsilon = epsilon
        self.rho = rho

    def transform(self, x, **kwargs):
        """
        Use DCT to transform the input image from spatial domain to frequency domain,
        Use IDCT to transform the input image from frequency domain to spatial domain.

        Arguments:
            x: (N, C, H, W) tensor for input images
        """
        gauss = torch.randn(x.size()[0], 3, 224, 224) * self.epsilon
        gauss = gauss.cuda()
        x_dct = self.dct_2d(x + gauss).cuda()
        mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
        high_frequency_mask = torch.zeros_like(mask)
        mask_h, mask_w = mask.shape[2], mask.shape[3]
        mask_ratio = 0.05
        low_freq_mask = torch.zeros_like(mask)
        low_freq_mask[:, :, :int(mask_h * mask_ratio), :int(mask_w * mask_ratio)] = 1
        high_frequency_mask = 1 - low_freq_mask

        x_idct = self.idct_2d(x_dct * mask*high_frequency_mask + x_dct * low_freq_mask)

        return x_idct

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

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            grads = 0
            for _ in range(self.num_spectrum):
                # Obtain the data after DCT and IDCT
                x_idct = self.transform(data + delta)

                # Obtain the output
                logits = self.get_logits(x_idct)

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients on x_idct
                grad = self.get_grad(loss, x_idct)
                grads += grad

            grads /= self.num_spectrum

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            X: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape).real

    def dct_2d(self, x, norm=None):
        """
        2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last 2 dimensions
        """
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        """
        The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct_2d(dct_2d(x)) == x
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            X: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last 2 dimensions
        """
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)




class SSM_P(MIFGSM):
    """
    Block-wise SSM Attack
    'Bag of tricks to boost the adversarial transferability'(https://arxiv.org/abs/2401.08734)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_spectrum (int): the number of spectrum.
        rho (float): the tuning factor for Uniform distribution.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, num_spectrum=20, rho=0.5

    Example script:
        python main.py --attack ssm --output_dir adv_data/ssm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., rho=0.5,num_scale=20,  targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None,  **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device)
        self.epsilon = epsilon
        self.num_scale = num_scale
        self.rho = rho
        self.num_block = 3
        self.ops = [self.scale, self.add_noise, self.drop_out]

    def transform(self, x, **kwargs):
        """
        Use DCT to transform the input image from spatial domain to frequency domain,
        Use IDCT to transform the input image from frequency domain to spatial domain.

        Arguments:
            x: (N, C, H, W) tensor for input images
        """
        gauss = torch.randn(x.size()[0], 3, 224, 224) * self.epsilon
        gauss = gauss.cuda()
        x_dct = self.dct_2d(x + gauss).cuda()
        mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
        high_frequency_mask = torch.zeros_like(mask)
        mask_h, mask_w = mask.shape[2], mask.shape[3]
        mask_ratio = 0.05
        high_frequency_mask[:, :, int(mask_h * mask_ratio):, int(mask_w * mask_ratio):] = 1
        mask = mask * high_frequency_mask
        x_idct = self.idct_2d(x_dct * mask)

        return x_idct

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

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            
            x_idct = self.transform(data + delta)
            # Obtain the output
            logits = self.get_logits(x_idct)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients on x_idct
            grad = self.get_grad(loss, delta)


            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def scale(self, x):
        return torch.rand(2)[0] * x 

    def add_noise(self, x):
        mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
        return x * mask

    def drop_out(self, x):
        return F.dropout2d(x, p=0.2, training=True)

    
    def dct_perturbation(self, x):
        gauss = torch.randn(x.size()[0], 3, 224, 224) * self.epsilon
        gauss = gauss.cuda()
        x_dct = self.dct_2d(x + gauss).cuda()
        x_dct = self.blocktransform(x_dct)
        x_idct = self.idct_2d(x_dct)

        return x_idct


    def blocktransform(self, x, choice=-1):
        x_copy = x.clone()
        _, _, w, h = x.shape
        ratio = 0.1
        pw = int(w * ratio)
        ph = int(h * ratio)

        chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.ops), dtype=np.int32)
        x_copy[:, :, pw:, ph:] = self.ops[chosen](x_copy[:, :, pw:, ph:])
        x_copy[:, :, :pw, ph:] = self.ops[chosen](x_copy[:, :, :pw, ph:])
        x_copy[:, :, pw:, :ph] = self.ops[chosen](x_copy[:, :, pw:, :ph])

        return x_copy

    def transform(self, x, **kwargs):
        return torch.cat([self.dct_perturbation(x) for _ in range(self.num_scale)], dim=0)

    def dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            X: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape).real

    def dct_2d(self, x, norm=None):
        """
        2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last 2 dimensions
        """
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        """
        The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct_2d(dct_2d(x)) == x
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Arguments:
            X: the input signal
            norm: the normalization, None or 'ortho'

        Return:
            the DCT-II of the signal over the last 2 dimensions
        """
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
    
    def get_loss(self, logits, label):
        return self.loss(logits, label.repeat(self.num_scale))
