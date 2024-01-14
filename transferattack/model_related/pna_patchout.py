import random
from functools import partial

from timm.models import create_model

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class PNA_PatchOut(MIFGSM):
    """
    PNA_PatchOut (Pay No Attention & PatchOut)
    'Towards Transferable Adversarial Attacks on Vision Transformers (AAAI 2022)'(https://arxiv.org/abs/2109.04176)

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
        ablation_study (str): the ablation study of the paper, including pna (pay no attention), patchout, l2 norm

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., ablation_study='1,1,1', crop_length=16, sample_num_patches=130, lamb=0.1

    Example scipt:
        python main.py --attack=pna_patchout
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, gamma=0.2, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='pna-patchout', ablation_study='1,1,1', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.model_name = 'vit_base_patch16_224'
        self.ablation_study = ablation_study.split(',')

        # pna (pay no attention)
        if self.ablation_study[0] == '1':
            print ('Using Skip')
            self._register_model()
        else:
            print ('Not Using Skip')

        # patchout
        self.image_size = 224
        self.crop_length = 16
        self.max_num_patches = int((224/16)**2)
        if self.ablation_study[1] == '1':
            print ('Using PatchOut')
            self.sample_num_patches = 130
        else:
            print ('Not Using PatchOut')
            self.sample_num_patches = self.max_num_patches
        assert self.sample_num_patches <= self.max_num_patches

        # l2 norm
        if self.ablation_study[2] == '1':
            print ('Using L2 Norm')
            self.lamb = 0.1
        else:
            print ('Not Using L2 Norm')
            self.lamb = 0

    def load_model(self, model_name):
        model = create_model(
                            model_name='vit_base_patch16_224',
                            pretrained=True,
                            num_classes=1000,
                            in_chans=3,
                            global_pool=None,
                            scriptable=False)
        model = wrap_model(model.eval().cuda())
        return model

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
        for epoch_idx in range(self.epoch):
            # Obtain the output
            delta_patchout = self._generate_samples_for_interactions(delta, epoch_idx) # use epoch_idx as seed
            logits = self.get_logits(self.transform(data+ delta_patchout, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Add l2 norm
            loss += self.lamb * torch.norm(delta, p=2)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def _register_model(self):
        """
        Register the backward hook for the attention dropout
        (This code is copied from https://github.com/zhipeng-wei/PNA-PatchOut)
        """
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model[1].blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        """
        Generate masked perturbations w.r.t. the patchout strategy
        (This code is copied from https://github.com/zhipeng-wei/PNA-PatchOut)
        """
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_patches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_patches])

        # Repeatable sampling
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation
