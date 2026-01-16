import os
import random
import torch
import torch.nn as nn
import numpy as np
import shap

from ..utils import *
from ..attack import Attack


class FAP(Attack):
    """
    Frequency-aware Perturbation (FAP) - Generation Version
    
    This is a generation-based implementation of the FAP attack that uses 
    frequency domain perturbations with pre-trained generators.
    
    Based on the original FAP attack from gradient/fap.py, but adapted for 
    the generation framework.
    
    Key features:
    - Uses YCbCr color space with 2D-DCT/IDCT transformations
    - 8x8 block-based frequency domain processing
    - Component subset selection via offline computed masks
    - Compatible with generation-based attack framework
    
    Parameters:
        model_name: Source model name
        epsilon: Base perturbation budget (default 8/255)
        alpha: Step size (calculated as epsilon'/epoch if None)
        epoch: Number of iterations (default 20)
        decay: Momentum decay factor (default 1.0)
        top_n: Number of frequency components to select (default 64)
        cache_dir: Directory for mask caching (default 'cache/fap_masks')
        bg_dir: Background data directory (default './data/images')
        bg_num: Number of background samples (default 100)
        block_size: Block size for frequency processing (default 8)
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/fap/resnet50 --attack fap --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/fap/resnet50 --eval
    """

    def __init__(
        self,
        model_name,
        epsilon=8/255,
        alpha=None,
        epoch=20,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm='linfty',
        loss='crossentropy',
        device=None,
        attack='FAP',
        top_n=64,
        cache_dir='cache/fap_masks',
        bg_dir='./path/to/data', #训练数据的目录
        bg_num=100,
        block_size=8,
        **kwargs,
    ):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.source_model_name = model_name
        self.base_epsilon = epsilon
        self.epoch = epoch
        self.decay = decay
        self.block = block_size
        self.top_n = int(top_n)
        self.bg_dir = bg_dir
        self.bg_num = int(bg_num)
        self.cache_dir = cache_dir

        self.epsilon = (self.base_epsilon * (192.0 / float(self.top_n)))
        self.alpha = (self.epsilon / self.epoch) if alpha is None else alpha

        self.freq_mask = None

    # ---------------------------- 核心前向 ---------------------------- #
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        self._ensure_freq_mask()

        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            delta_prime = self.apply_frequency_gate_to_delta(delta)
            x_adv = (data + delta_prime).detach().requires_grad_(True)

            logits = self.get_logits(x_adv)
            loss = self.get_loss(logits, label)
            grad_x = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            momentum = self.get_momentum(grad_x, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    # ------------------------ frequency_gate delta ------------------------ #
    def apply_frequency_gate_to_delta(self, delta: torch.Tensor) -> torch.Tensor:
        # RGB -> YCbCr
        ycbcr = self.rgb_to_ycbcr(delta)

        # DCT
        v = self._dct_2d_safe(ycbcr)

        # blockify: (B,C,H,W) -> (B,C,N,8,8)
        u = self.blockify(v, self.block)

        mask = self.freq_mask.to(u.device).float().view(1, 3, 1, self.block, self.block)
        u_gated = u * mask

        v_gated = self.deblockify(u_gated, self.block)
        x_spatial = self._idct_2d_safe(v_gated)
        rgb = self.ycbcr_to_rgb(x_spatial)

        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb

    # ------------------------ mask ------------------------ #
    def _ensure_freq_mask(self):
        if self.freq_mask is not None:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"mask_top{self.top_n}.pt")
        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location='cpu')
            self.freq_mask = data['mask'].bool()
            return

        print(f"=> Building FAP frequency mask via SHAP (top {self.top_n}) from {self.bg_dir}, num={self.bg_num}")
        try:
            mask = self._build_frequency_mask_via_shap(self.bg_dir, self.bg_num, self.block)
        except Exception as e:
            print(f"[FAP] SHAP building failed ({e}), falling back to gradient-based importance.")
            mask = self._build_frequency_mask_via_grad(self.bg_dir, self.bg_num, self.block)
        torch.save({'mask': mask.cpu(), 'top_n': self.top_n}, cache_path)
        self.freq_mask = mask

    def _iter_bg_loader(self, bg_dir: str, batch_size: int = 16):
        root = self._resolve_dataset_root(bg_dir)
        dataset = AdvDataset(input_dir=root, targeted=False, eval=False)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: self.bg_num]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        for images, labels, _ in loader:
            yield images.to(self.device), labels.to(self.device)

    def _build_frequency_mask_via_grad(self, bg_dir: str, bg_num: int, block: int) -> torch.Tensor:
        accum = torch.zeros(3, block, block, device=self.device)
        count = 0

        for images, labels in self._iter_bg_loader(bg_dir):
            images = images.clone().requires_grad_(True)
            logits = self.get_logits(images)
            loss = self.get_loss(logits, labels)
            grad_x = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

            grad_ycbcr = self.rgb_to_ycbcr(grad_x)
            grad_freq = self._dct_2d_safe(grad_ycbcr)

            g_u = self.blockify(grad_freq, block)  # (B,C,N,8,8)
            g_map = g_u.abs().mean(dim=2)  # (B,C,8,8)

            accum += g_map.mean(dim=0)
            count += 1

            if count * images.size(0) >= self.bg_num:
                break

        accum = accum / max(count, 1)

        flat = accum.view(-1)
        topk = torch.topk(flat, k=self.top_n, largest=True).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[topk] = True
        mask = mask.view(3, block, block)
        return mask

    def _build_frequency_mask_via_shap(self, bg_dir: str, bg_num: int, block: int) -> torch.Tensor:
        shap_model = self.load_model(self.source_model_name)
        self._make_model_non_inplace(shap_model)

        root = self._resolve_dataset_root(bg_dir)
        dataset = AdvDataset(input_dir=root, targeted=False, eval=False)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: bg_num]

        bg_count = min(20, len(indices))
        bg_imgs = []
        for i in range(bg_count):
            img, _, _ = dataset[indices[i]]
            bg_imgs.append(img)
        background = torch.stack(bg_imgs, dim=0).to(self.device)

        self.model.eval()
        explainer = None
        try:
            explainer = shap.DeepExplainer(shap_model, background)
        except Exception:
            explainer = shap.GradientExplainer(shap_model, background)

        accum = torch.zeros(3, block, block, device=self.device)
        processed = 0

        eval_indices = indices
        batch_size = 8
        for start in range(0, len(eval_indices), batch_size):
            end = min(start + batch_size, len(eval_indices))
            batch_imgs = []
            batch_labels = []
            for j in range(start, end):
                img, label, _ = dataset[eval_indices[j]]
                batch_imgs.append(img)
                batch_labels.append(label)
            x = torch.stack(batch_imgs, dim=0).to(self.device)
            y = torch.tensor(batch_labels, device=self.device)

            try:
                shap_vals_list = explainer.shap_values(x, check_additivity=False)
            except Exception:
                explainer = shap.GradientExplainer(shap_model, background)
                shap_vals_list = explainer.shap_values(x, check_additivity=False)

            if isinstance(shap_vals_list, list):
                sv = torch.from_numpy(np.stack(shap_vals_list, axis=0)).to(self.device)
                b = x.size(0)
                gather_idx = y.view(1, b, 1, 1, 1).expand(1, b, sv.size(2), sv.size(3), sv.size(4)).long()
                sv_y = sv.gather(0, gather_idx).squeeze(0)  # (B,C,H,W)
            else:
                sv_y = torch.from_numpy(shap_vals_list).to(self.device)

            sv_y = sv_y.float()
            sv_ycbcr = self.rgb_to_ycbcr(sv_y)
            sv_freq = self._dct_2d_safe(sv_ycbcr)
            sv_u = self.blockify(sv_freq, block)  # (B,C,N,8,8)
            sv_map = sv_u.abs().mean(dim=2)      # (B,C,8,8)
            accum += sv_map.mean(dim=0)

            processed += x.size(0)
            if processed >= bg_num:
                break

        accum = accum / max(processed // max(1, batch_size), 1)

        flat = accum.view(-1)
        topk = torch.topk(flat, k=self.top_n, largest=True).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[topk] = True
        mask = mask.view(3, block, block)
        del explainer
        del shap_model
        torch.cuda.empty_cache()
        return mask

    def _resolve_dataset_root(self, path: str) -> str:
        p = path.rstrip('/')
        if os.path.isdir(os.path.join(p, 'images')) and os.path.isfile(os.path.join(p, 'labels.csv')):
            return p
        if os.path.basename(p) == 'images':
            return os.path.dirname(p)
        return p

    def _make_model_non_inplace(self, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU) and getattr(child, 'inplace', False):
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                self._make_model_non_inplace(child)

    def _dct_2d_safe(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        try:
            return self.dct_2d(x)
        except Exception:
            x_cpu = x.detach().cpu()
            v_cpu = self.dct_2d(x_cpu)
            return v_cpu.to(dev)

    def _idct_2d_safe(self, X: torch.Tensor) -> torch.Tensor:
        dev = X.device
        try:
            return self.idct_2d(X)
        except Exception:
            X_cpu = X.detach().cpu()
            x_cpu = self.idct_2d(X_cpu)
            return x_cpu.to(dev)

    def rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:2+1, :, :]
        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat([y, cb, cr], dim=1)

    def ycbcr_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        y, cb, cr = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1)

    def blockify(self, x: torch.Tensor, size: int):
        b, c, h, w = x.shape
        assert h % size == 0 and w % size == 0
        x = x.view(b, c, h // size, size, w // size, size)  # (b,c,hy,s,wy,s)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()        # (b,c,hy,wy,s,s)
        x = x.view(b, c, (h // size) * (w // size), size, size)
        return x

    def deblockify(self, x: torch.Tensor, size: int):
        b, c, n, s1, s2 = x.shape
        H = img_height
        W = img_width
        hy = H // size
        wy = W // size
        x = x.view(b, c, hy, wy, size, size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c, hy * size, wy * size)
        return x

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = torch.fft.fft(v)
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)
        return V

    def idct(self, X, norm=None):
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
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)


