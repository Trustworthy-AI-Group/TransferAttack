import torch

from ..utils import *
from ..attack import Attack
import scipy.stats as st
from torch.autograd import Variable 
import torch.nn.functional as F

class FGSRA(Attack):
    """
    FGSRA(Frequency-Guided Sample Relevance Attack)
    'Improving Adversarial Transferability via Frequency-Guided Sample Relevance Attack (CIKM2024)'(https://dl.acm.org/doi/10.1145/3627673.3679858) 

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        rho (float): frequency tuning factor.
        beta (float): the neighborhood range factor.
        max_iter (int): the number of neighboring samples.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, rho=0.7, beta=2.0, max_iter=20

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/fgsra/resnet50 --attack fgsra --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/fgsra/resnet50 --eval   
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, rho=0.7, beta=2.0, max_iter=20, epoch=10, decay=1., targeted=False,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='FGSRA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.rho = rho
        self.beta = beta
        self.max_iter = max_iter
        self.targeted = False
        
    def dct(self, x, norm=None):
        """
        Discrete Cosine Transform (DCT)
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
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform
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
        2-dimentional Discrete Cosine Transform (DCT)
        """
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)
    
    def idct_2d(self, x, norm=None):
        """
        2-dimentional Inverse Discrete Cosine Transform (IDCT)
        """
        X1 = self.idct(x, norm=norm)
        X2 = self.idct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)
    
    def gkern(self, kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
        return gaussian_kernel
    
    def DI_Resize(self, x, resize_rate=1.15, diversity_prob=0.5):
        assert resize_rate >= 1.0
        assert diversity_prob >= 0.0 and diversity_prob <= 1.0
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        ret = padded if torch.rand(1) < diversity_prob else x
        ret = F.interpolate(ret, size=[img_size, img_size], mode='bilinear', align_corners=False)
        return ret
    
    def forward(self, data, label, **kwargs):
        
        """
        The attack procedure for FGSRA
        
        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        # T_kernel = self.gkern(7, 3)
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        delta = self.init_delta(data)

        m = torch.ones_like(data)*10 / 9.4
        momentum = 0
        
        for _ in range(self.epoch): 
            x = data+delta   
            logits = self.get_logits(x, momentum=momentum)
            loss = self.get_loss(logits, label)
            current_grad = self.get_grad(loss, delta)
            avg_grad = list()
            x_sims = list()
            
            for _ in range(self.max_iter):
                gauss = torch.rand_like(x)*2*(self.epsilon * self.beta) - self.epsilon * self.beta
                gauss = gauss.to(self.device)
                x_dct = self.dct_2d(x+gauss).to(self.device)
                mask  = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).to(self.device)
                x_idct= self.idct_2d(x_dct * mask)
                x_idct.requires_grad_(True)

                logits_i = self.get_logits(x_idct)
                loss_i = self.get_loss(logits_i, label)
                grad_i = self.get_grad(loss_i, delta)
                avg_grad.append(grad_i)
                cossim = (x * x_idct).sum([1,2,3], keepdim=True) / (torch.sqrt((x ** 2).sum([1, 2, 3], keepdim=True)) * torch.sqrt((x_idct ** 2).sum([1, 2, 3], keepdim=True)))
                x_sims.append(cossim)
 
            x_sims = torch.stack(x_sims, dim=1)
            avg_grad = torch.stack(avg_grad, dim=1)
            avg_grad = (avg_grad * x_sims).sum(1)
            
            cossim = (current_grad * avg_grad).sum([1, 2, 3], keepdim=True) / (
                        torch.sqrt((current_grad ** 2).sum([1, 2, 3], keepdim=True)) * torch.sqrt(
                    (avg_grad ** 2).sum([1, 2, 3], keepdim=True)))
            current_grad = cossim * current_grad + (1-cossim) * avg_grad
            
            #* current_grad = F.conv2d(current_grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
            
            momentum = self.get_momentum(current_grad, momentum)
            eqm = (torch.sign(momentum) == torch.sign(current_grad)).float() 
            neqm = torch.ones_like(data) - eqm
            m = m * (eqm + neqm * 0.94)
            delta = self.update_delta(delta, data, momentum, self.alpha*m)
        
        return delta.detach()
                