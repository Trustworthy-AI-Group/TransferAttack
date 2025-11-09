from timm.layers import mixed_conv2d
import torch
import torch.fft
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class MFI(MIFGSM):
    """
    MFI Attack (Mixed-Frequency Inputs)
    'Enhancing Transferability of Adversarial Examples Through Mixed-Frequency Inputs (TIFS 2024)' (https://ieeexplore.ieee.org/document/10602524)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_sample (int): the number of mixed images to sample.
        num_scale (int): the number of scales to apply.
        mask_radius (int): the mask filter radius r (default: 100).
        gaussian_sigma (float): the strength of Gaussian noise perturbation σ (default: 32).
        mfi_type (str): type of MFI to use - 'hmfi' for Hard MFI or 'smfi' for Soft MFI (default: 'smfi').
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
    
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., mask_radius=100, gaussian_sigma=32, mfi_type='smfi'

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mfi/resnet50 --attack mfi --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/mfi/resnet50 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,
                 mask_radius=100, gaussian_sigma=32, mfi_type='smfi', targeted=False, 
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='MFI', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.mask_radius = mask_radius
        self.gaussian_sigma = gaussian_sigma
        self.num_sample = 3
        self.num_scale = 5
        self.mfi_type = mfi_type  # 'hmfi' or 'smfi'

        # cache masks keyed by (H,W,device,softness_flag)
        self._mask_cache = {}

    # ---------------------------
    # Helper utilities
    # ---------------------------
    def _get_mask(self, H, W, device, soft=False, softness=10.0):
        key = (H, W, str(device), soft, float(softness), int(self.mask_radius))
        if key in self._mask_cache:
            return self._mask_cache[key]
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        cy, cx = H // 2, W // 2
        dist = torch.sqrt((y - cy).float()**2 + (x - cx).float()**2)
        if not soft:
            mask = (dist <= self.mask_radius).float()
        else:
            mask = torch.sigmoid(-(dist - self.mask_radius) / softness)
        mask = mask.view(1, 1, H, W)  # (1,1,H,W) for broadcasting
        self._mask_cache[key] = mask
        return mask

    def _fft(self, x):
        # expects real spatial x: shape (B,C,H,W) or (m,C,H,W)
        X = torch.fft.fft2(x, dim=(-2, -1))
        return torch.fft.fftshift(X, dim=(-2, -1))

    def _ifft(self, X):
        X = torch.fft.ifftshift(X, dim=(-2, -1))
        return torch.fft.ifft2(X, dim=(-2, -1)).real

    # ---------------------------
    # Transform implementations
    # ---------------------------
    def apply_hmfi(self, x, mix_images):
        """
        Hard Mixed-Frequency Inputs (HMFI): Simple hard frequency mixing
        - Keep low frequency from input x
        - Replace high frequency with high frequency from mix images
        """
        B, C, H, W = x.shape

        # Create hard frequency mask
        mask_l = self._get_mask(H, W, self.device, soft=False)  # low-pass (center True)
        mask_h = 1.0 - mask_l                               # high-pass

        # FFT
        X = self._fft(x)            # (B,C,H,W)
        M = self._fft(mix_images)   # (m,C,H,W)

        # HMFI: Keep low freq from x, high freq from mix images
        mixed_fft = X * mask_l + M * mask_h

        out = self._ifft(mixed_fft)
        
        return out

    def apply_smfi(self, x, mix_images):
        """
        Soft Mixed-Frequency Inputs (SMFI): Soft frequency mixing with smooth transitions
        - Use soft mask for smooth frequency transitions
        - Keep low frequency from input x
        - Mix high frequency from input and mix images
        """
        B, C, H, W = x.shape
        # Create soft frequency mask
        mask_l = self._get_mask(H, W, self.device, soft=True, softness=10.0)
        mask_h = 1.0 - mask_l

        # FFT
        X = self._fft(x)            # (B,C,H,W)
        M = self._fft(mix_images)   # (m,C,H,W)

        # Randomly select mix images for each input
        # SMFI: Soft mixing - blend high frequencies
        high_freq_x = X * mask_h
        high_freq_mix = M * mask_h
        
        # Soft interpolation between high frequencies
        alpha = torch.rand(1).to(self.device)
        high_freq_blend = alpha * high_freq_x + (1 - alpha) * high_freq_mix

        # Combine low freq from x with blended high freq
        mixed_fft = X * mask_l + high_freq_blend

        out = self._ifft(mixed_fft)

        return out

    # ---------------------------
    # Public transform API
    # ---------------------------
    def get_mixed_images(self, x, **kwargs):
        """
        Apply Mixed-Frequency Input transformation.
        - With probability (1 - diversity_prob), bypass and return x (no transform).
        - Uses in-batch mix images if self.mix_images is None; otherwise uses provided pool.
        - Returns transformed images in same dtype/device, clamped to [0,1].
        """
        pool = x[torch.randperm(x.size(0))].detach()

        # choose which MFI to apply
        if self.mfi_type == 'hmfi':
            mixed = self.apply_hmfi(x, pool)
        else:
            mixed = self.apply_smfi(x, pool)

        # add gaussian noise
        mixed += torch.randn_like(x) * (self.gaussian_sigma / 255.0)

        return mixed

    def transform(self, x, **kwargs):
        """
        Admix the input for Admix Attack
        """
        mixed_images = torch.concat([self.get_mixed_images(x) for _ in range(self.num_sample)], dim=0)
        return torch.concat([mixed_images / (2 ** i) for i in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale*self.num_admix)) if self.targeted else self.loss(logits, label.repeat(self.num_scale*self.num_admix))
