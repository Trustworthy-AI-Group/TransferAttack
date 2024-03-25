from functools import partial

import torch

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class TGR(MIFGSM):
    """
    TGR (Token Gradient Regularization)
    'Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization (CVPR 2023)'(https://arxiv.org/abs/2303.15754)

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

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.0, mlp_gamma=0.25 (we follow mlp_gamma=0.5 in official code)

    Example script:
        python main.py --attack=tgr --input_dir=./data --output_dir=./results/tgr/vit --model vit_base_patch16_224 --batchsize 1

    NOTE:
        1) The code only support batchsize = 1.
    """


    def __init__(self, **kwargs):
        self.model_name = kwargs['model_name']
        kwargs['attack'] = 'TGR'
        super().__init__(**kwargs)

        self.model = self.model[1] # unwrap the model
        self._register_model()
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again



    def _register_model(self):
        """
        Copied from https://github.com/jpzhang1810/TGR/blob/master/methods.py
        """
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in ['vit_base_patch16_224', 'visformer_small', 'pit_b_224']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,:] = 0.0
                out_grad[:,range(C),:,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,:] = 0.0
                out_grad[:,range(C),:,min_all_W] = 0.0

            if self.model_name in ['cait_s24_224']:
                B,H,W,C = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, H*W, C)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                min_all_H = min_all//H
                min_all_W = min_all%H

                out_grad[:,max_all_H,:,range(C)] = 0.0
                out_grad[:,:,max_all_W,range(C)] = 0.0
                out_grad[:,min_all_H,:,range(C)] = 0.0
                out_grad[:,:,min_all_W,range(C)] = 0.0

            return (out_grad, )

        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            B,H,W,C = grad_in[0].shape
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0,:,0,:], axis = 0)
            min_all = np.argmin(out_grad_cpu[0,:,0,:], axis = 0)

            out_grad[:,max_all,:,range(C)] = 0.0
            out_grad[:,min_all,:,range(C)] = 0.0
            return (out_grad, )

        def q_tgr(module, grad_in, grad_out, gamma):
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad[:] = 0.0
            return (out_grad, grad_in[1], grad_in[2])

        def v_tgr(module, grad_in, grad_out, gamma):
            # show diff between high and low PyTorch version
            # print('v len(grad_in)',len(grad_in))
            # high, 1
            # low, 2

            # print('v grad_in[0].shape',grad_in[0].shape)
            # high, torch.Size([197, 2304])
            # low, torch.Size([1, 197, 2304])
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.model_name in ['visformer_small']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,min_all_W] = 0.0

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)

                out_grad[:,max_all,range(c)] = 0.0
                out_grad[:,min_all,range(c)] = 0.0

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)

            # return (out_grad, grad_in[1])
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics

        def mlp_tgr(module, grad_in, grad_out, gamma):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in ['visformer_small']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,min_all_W] = 0.0
            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'resnetv2_101']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                out_grad[:,max_all,range(c)] = 0.0
                out_grad[:,min_all,range(c)] = 0.0

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)

            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics


        attn_tgr_hook = partial(attn_tgr, gamma=0.25)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.25)
        v_tgr_hook = partial(v_tgr, gamma=0.75)
        q_tgr_hook = partial(q_tgr, gamma=0.75)

        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook)
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
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(attn_cait_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.q.register_backward_hook(q_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.k.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.v.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_tgr_hook)
