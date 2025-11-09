from collections import OrderedDict

from ..utils import *
from ..attack import Attack


class MFAA(Attack):
    """
    MFAA (Multi-Feature Attention Attack)
    'Enhancing the Transferability of Adversarial  Attacks via Multi-Feature Attention (TIFS 2025)' (https://ieeexplore.ieee.org/document/10833658)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_ens (int): the number of ensemble for guidance calculation.
        probb (float): the keep probability for random masking.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, probb=0.8, num_ens=30, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mfaa/resnet50 --attack mfaa --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/mfaa/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, num_ens=30, probb=0.8,
                 targeted=False,random_start=False,norm='linfty',loss='crossentropy', device=None, attack='MFAA'):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.epoch = int(epoch)
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.num_ens = int(num_ens)
        self.probb   = float(probb)

        self._init_layer_hooks()

    def _init_layer_hooks(self):
        # get model name
        depths = [len(self.model[1].layer1), len(self.model[1].layer2), len(self.model[1].layer3), len(self.model[1].layer4)]

        if depths == [3, 8, 36, 3]: 
            # ResNet-152: [3, 8, 36, 3]
            try:
                self._layers = OrderedDict({
                    "L4_u3":  self.model[1].layer4[-1],   # block4/unit_3
                    "L3_u29": self.model[1].layer3[-8],   # block3/unit_29
                    "L3_u19": self.model[1].layer3[-17],  # block3/unit_19
                    "L3_u9":  self.model[1].layer3[-27],  # block3/unit_9
                    "L2_u7":  self.model[1].layer2[-1],   # block2/unit_7 (final attacked layer)
                })
            except Exception as e:
                raise RuntimeError(
                    "MFAA expects a ResNet-152-like backbone with .layer{1..4}. "
                    "Adjust taps if your surrogate differs.\n" + str(e)
                )
        elif depths == [3, 4, 6, 3]:
            # ResNet-50: [3, 4, 6, 3] - map to equivalent relative positions
            try:
                self._layers = OrderedDict({
                    "L4_u3":  self.model[1].layer4[-1],   # block4/unit_3 (last unit)
                    "L3_u5":  self.model[1].layer3[-1],   # block3/unit_5 (last unit, equivalent to unit_29 in ResNet-152)
                    "L3_u3":  self.model[1].layer3[-3],   # block3/unit_3 (middle unit, equivalent to unit_19)
                    "L3_u1":  self.model[1].layer3[-5],   # block3/unit_1 (early unit, equivalent to unit_9)
                    "L2_u3":  self.model[1].layer2[-1],   # block2/unit_3 (last unit, equivalent to unit_7)
                })
            except Exception as e:
                raise RuntimeError(
                    "Adjust taps if your surrogate differs.\n" + str(e)
                )
        else:
            raise RuntimeError(f"Unsupported ResNet architecture with depths {depths}. Expected [3,8,36,3] for ResNet-152 or [3,4,6,3] for ResNet-50")
            
        self._mid_outputs = {k: None for k in self._layers}
        self._handles     = []

    def _mk_fwd_hook(self, name):
        def _f(m, i, o):
            self._mid_outputs[name] = o
        return _f

    def _register_forward_hooks(self):
        self._handles += [mod.register_forward_hook(self._mk_fwd_hook(n)) for n, mod in self._layers.items()]

    def _remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    @staticmethod
    def _l2_normalize_per_sample(t, eps: float = 1e-12):
        b = t.shape[0]
        v = t.reshape(b, -1)
        n = torch.sqrt((v * v).sum(dim=1, keepdim=True) + eps)
        return (v / n).reshape_as(t)

    @staticmethod
    def _fia_loss_adv_only(fmap_2b, weights, B):
        """
        EA loss on ADV half only: sum(adv * weights)/numel.
        fmap_2b: [2B, C, H, W], weights: [B, C, H, W]
        """
        adv = fmap_2b[B:]
        numel = float(fmap_2b.numel())
        return (adv * weights).sum() / numel

    def _drop_mask(self, x):
        # Keep probability self.probb (drop prob = 1 - probb)
        mask = torch.bernoulli(torch.ones_like(x) * self.probb).to(x.device)
        x_drop = (x * mask).detach()
        x_drop.requires_grad_(True)
        return x_drop

    # ---------------- guidance ON CLEAN (once at i==0) ----------------

    def _compute_guidance_on_clean(self, data, y_sel):
        B = data.shape[0]
        accum = {k: None for k in self._layers}

        # Temporarily register hooks for guidance computation
        temp_handles = []
        for name, module in self._layers.items():
            def make_hook(n):
                def hook(m, i, o):
                    self._mid_outputs[n] = o
                return hook
            temp_handles.append(module.register_forward_hook(make_hook(name)))
        
        try:
            for _ in range(max(1, self.num_ens)):
                x_drop = self._drop_mask(data)                     # [B,C,H,W]
                x_cat  = torch.cat([x_drop, x_drop], dim=0)       # [2B,...]
                y_cat  = torch.cat([y_sel, y_sel], dim=0)         # [2B]

                logits = self.model(x_cat)
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, y_cat.view(-1, 1), 1)
                chosen = (logits * y_onehot).sum()

                # grads w.r.t. each tapped fmap
                grads_this = {}
                for k, fmap in self._mid_outputs.items():
                    if fmap is not None:
                        g_full = torch.autograd.grad(chosen, fmap, retain_graph=True, allow_unused=True)[0]
                        if g_full is not None:
                            grads_this[k] = g_full[B:].detach()        # ADV half only

                for k, g in grads_this.items():
                    accum[k] = g if accum[k] is None else (accum[k] + g)
        finally:
            # Remove temporary hooks
            for handle in temp_handles:
                handle.remove()

        weights = {}
        for k, g in accum.items():
            if g is not None:
                g = g / max(1, self.num_ens)
                weights[k] = -self._l2_normalize_per_sample(g)     # NEGATE like TF
            else:
                # Fallback: create zero weights if no gradients were computed
                weights[k] = torch.zeros_like(data)
        return weights

    # --------------------------- main ---------------------------

    def forward(self, data, label, **kwargs):
        """
        Args:
            data:  (N,C,H,W) tensor.
            label: (N,) long tensor OR [gt, tgt] when targeted. We follow Attack semantics:
                   if self.targeted: use label[1] as selected label; else use label.
        """
        # Handle targeted/untargeted label selection aligned with Attack semantics
        if self.targeted and isinstance(label, (list, tuple)) and len(label) == 2:
            y_sel = label[1]
        else:
            y_sel = label

        data  = data.clone().detach().to(self.device)
        y_sel = y_sel.clone().detach().to(self.device)

        # init delta & momentum buffer
        delta = self.init_delta(data)
        momentum_buf = torch.zeros_like(data, device=self.device)
        self._register_forward_hooks()

        # Compute guidance on clean data (once) - this needs gradients
        guidance = self._compute_guidance_on_clean(data, y_sel)

        for _ in range(self.epoch):
            x_adv = (data + delta).detach()
            x_adv.requires_grad_(True)

            # Forward on concat(clean, adv) to get fmaps for both halves
            x_cat = torch.cat([data.detach(), x_adv], dim=0)  # [2B,...]
            _ = self.model(x_cat)

            B      = data.shape[0]
            f_L4   = self._mid_outputs["L4_u3"]
            f_L3_1 = self._mid_outputs["L3_u5"]   # Updated for ResNet-50
            f_L3_2 = self._mid_outputs["L3_u3"]   # Updated for ResNet-50
            f_L3_3 = self._mid_outputs["L3_u1"]   # Updated for ResNet-50
            f_L2   = self._mid_outputs["L2_u3"]   # Updated for ResNet-50

            # Validate that all feature maps are available
            if any(f is None for f in [f_L4, f_L3_1, f_L3_2, f_L3_3, f_L2]):
                raise RuntimeError("Some feature maps are None. Check layer hook registration.")

            # LAG chain mirrors TF:
            # loss at L4 with guidance[L4], then propagate grads to L3_1 (adv half), normalize, add guidance[L3_1], etc.
            loss_L4   = self._fia_loss_adv_only(f_L4,   guidance["L4_u3"],  B)

            g_L3_1_f  = torch.autograd.grad(loss_L4,  f_L3_1, retain_graph=True)[0]
            g_L3_1    = g_L3_1_f[B:].detach()
            w_L3_1    = self._l2_normalize_per_sample(g_L3_1) + guidance["L3_u5"]   # Updated for ResNet-50
            loss_L3_1 = self._fia_loss_adv_only(f_L3_1, w_L3_1, B)

            g_L3_2_f  = torch.autograd.grad(loss_L3_1, f_L3_2, retain_graph=True)[0]
            g_L3_2    = g_L3_2_f[B:].detach()
            w_L3_2    = self._l2_normalize_per_sample(g_L3_2) + guidance["L3_u3"]   # Updated for ResNet-50
            loss_L3_2 = self._fia_loss_adv_only(f_L3_2, w_L3_2, B)

            g_L3_3_f  = torch.autograd.grad(loss_L3_2, f_L3_3, retain_graph=True)[0]
            g_L3_3    = g_L3_3_f[B:].detach()
            w_L3_3    = self._l2_normalize_per_sample(g_L3_3) + guidance["L3_u1"]   # Updated for ResNet-50
            loss_L3_3 = self._fia_loss_adv_only(f_L3_3, w_L3_3, B)

            g_L2_f    = torch.autograd.grad(loss_L3_3, f_L2,   retain_graph=True)[0]
            g_L2      = g_L2_f[B:].detach()
            w_L2      = self._l2_normalize_per_sample(g_L2) + guidance["L2_u3"]     # Updated for ResNet-50
            loss   = self._fia_loss_adv_only(f_L2, w_L2, B)   # final objective at block2/unit_3

            # Grad wrt x_adv and MI-FGSM ascend (+momentum), then clamp
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            momentum_buf = self.get_momentum(grad, momentum_buf)

            delta = self.update_delta(delta, data, momentum_buf, self.alpha)  

        self._remove_hooks()
        return delta.detach()
