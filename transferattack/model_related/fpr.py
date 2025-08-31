from functools import partial
import torch
import random
from ..gradient.mifgsm import MIFGSM
from ..utils import *

accumulated_features = {}

class FPR(MIFGSM):
    """
    FPR Attack
    'Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement (CVPR 2025)'(https://arxiv.org/pdf/2503.15404)

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
        device (torch.device): the device for data. If it is None, the device would be same as model.

    Official arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/fpr/vit --attack fpr --model=vit_base_patch16_224 --batchsize=8
        python main.py --input_dir ./path/to/data --output_dir adv_data/fpr/vit --eval

    Notes:
        Batch size should be set to: batchsize=8

    """

    def __init__(self, model_name='vit_base_patch16_224', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False,  random_start=False, norm='linfty', loss='crossentropy', device=None, 
                 attack='fpr', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        self.model_name = model_name
        self.model = self.model[1]

        self._register_model()
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again

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

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def _register_model(self):
        def diverse_attn_map(module, input, output, attn_map_change_range):
            batch_size, num_heads, seq_length, _ = output.shape
            s_output = output * 1.0
            attn_map_noise = torch.empty_like(s_output)
            for head in range(num_heads):
                M = torch.tensor(np.random.uniform(1 - attn_map_change_range, 1 + attn_map_change_range, (seq_length, seq_length)), dtype=torch.float32).to(output.device)
                noisy_attn = s_output[:, head, :, :] * M  
                normalized_attn = torch.softmax(noisy_attn, dim=-1)
                attn_map_noise[:, head, :, :] = normalized_attn

            return attn_map_noise

        def cross_iter_emb_momentum(module, input, output, scale, mom_emb_decay): 
            s = scale 
            s_output = output * s
            module_id = id(module)
            if module_id not in accumulated_features:
                accumulated_features[module_id] = s_output.clone()
            else:
                accumulated_features[module_id] = mom_emb_decay * accumulated_features[module_id].clone().detach() + s_output

            return accumulated_features[module_id]
        
        if self.model_name in ['vit_base_patch16_224' ]:
            for i in [0,1,4,9,11]: 
                self.model.blocks[i].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=25))
            for i in range(12):
                self.model.blocks[i].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=0.8, mom_emb_decay=0.3))
            for i in range(12):
                self.model.blocks[i].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=0.8, mom_emb_decay=0.3))
