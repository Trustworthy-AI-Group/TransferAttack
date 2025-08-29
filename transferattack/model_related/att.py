from functools import partial

import torch
import random
from ..gradient.mifgsm import MIFGSM
from ..utils import *


class ATT(MIFGSM):
    """
    ATT (Adaptive Token Tuning)
    'Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning (NeurIPS 2024)'(https://proceedings.neurips.cc/paper_files/paper/2024/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html)

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
        lam (float): the adaptive factor.
        gamma (float): the gradient penalty factor.
        sample_num_batches (int): the number of PatchOut discards.
        scale (float): the scaling factor.
        offset (float): the offset factor.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.0, lam=0.01, gamma=0.5, scale=0.4, offset=0.4

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/att/vit --attack=att --model vit_base_patch16_224 --batchsize 1
        python main.py --input_dir ./path/to/data --output_dir adv_data/att/vit --eval

    """

    def __init__(self, model_name='vit_base_patch16_224',  epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., 
                 targeted=False,  random_start=False, norm='linfty', 
                 loss='crossentropy', device=None, attack='att', 
                 lam=0.01, sample_num_batches=130, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = 1.
        
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16) ** 2)
        
        self.model_name = model_name
        self.im_fea = None
        self.im_grad = None
        self.size = 16
        self.lam = lam
        self.patch_index = self.Patch_index(self.size)
        
        self.model = self.model[1] # unwrap the model
        self._register_model()
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again

    def TR_01_PC(self, num, length):
        rate_l = num
        tensor = torch.cat((torch.ones(rate_l), torch.zeros(length - rate_l)))
        return tensor

    def _register_model(self):
        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        self.gamma = 0.5
        self.back_attn = 11
        self.truncate_layers = self.TR_01_PC(10, 12)
        self.weaken_factor =  [0.45, 0.7, 0.65]
        self.scale = 0.4
        self.offset = 0.4 
    
        def attn_ATT(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.gamma
            out_grad = mask * grad_in[0][:]
            if self.var_A != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_A))).clamp(0, 1)
            else:
                GPF_ = self.gamma            
            
            if self.model_name in ['vit_base_patch16_224', 'visformer_small', 'pit_b_224']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,:] *= GPF_
                out_grad[:,range(C),:,max_all_W] *= GPF_
                out_grad[:,range(C),min_all_H,:] *= GPF_
                out_grad[:,range(C),:,min_all_W] *= GPF_
            
            self.var_A = torch.var(out_grad)
            self.back_attn -= 1
            return (out_grad, )

        def q_ATT(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            if self.var_qkv != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            out_grad[:] *= GPF_
            self.var_qkv = torch.var(out_grad)
            return (out_grad, grad_in[1], grad_in[2])

        def v_ATT(module, grad_in, grad_out):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            
            if self.var_qkv != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF_ = self.gamma
                
            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)

                out_grad[:, max_all, range(c)] *= GPF_
                out_grad[:, min_all, range(c)] *= GPF_
            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            self.var_qkv = torch.var(out_grad)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics

        def mlp_ATT(module, grad_in, grad_out):
            if len(grad_in[0].shape) == 2:
                # print("len(grad_in[0].shape) == 2")
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)
            
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
            out_grad = mask * grad_in[0][:]
            if self.var_mlp != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_mlp))).clamp(0, 1)
            else:
                GPF_ = self.gamma

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'resnetv2_101']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                out_grad[:,max_all,range(c)] *= GPF_
                out_grad[:,min_all,range(c)] *= GPF_

            self.var_mlp = torch.var(out_grad)
            out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics
        
        def get_fea(module, input, output):
            self.im_fea = output.clone()

        def get_grad(module, input, output):
            self.im_grad = output[0].clone()


        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224']:
            self.get_fea_hook = self.model.blocks[10].register_forward_hook(get_fea)
            self.get_grad_hook = self.model.blocks[10].register_backward_hook(get_grad)
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_ATT)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_ATT)
                self.model.blocks[i].mlp.register_backward_hook(mlp_ATT)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size / self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:, :, r * self.crop_length:(r + 1) * self.crop_length,
            c * self.crop_length:(c + 1) * self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation
    
    def Patch_index(self, size):
        img_size = 224
        filterSize = size
        stride = size
        P = np.floor((img_size - filterSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P
        index = np.ones([P * Q, filterSize * filterSize], dtype=int)
        tmpidx = 0
        for q in range(Q):
            plus1 = q * stride * img_size
            for p in range(P):
                plus2 = p * stride
                index_ = np.array([], dtype=int)
                for i in range(filterSize):
                    plus = i * img_size + plus1 + plus2
                    index_ = np.append(index_, np.arange(plus, plus + filterSize, dtype=int))
                index[tmpidx] = index_
                tmpidx += 1
        index = torch.LongTensor(np.tile(index, (1, 1, 1))).cuda()
        return index
    
    def norm_patchs(self, GF, index, patch, scale, offset):
        patch_size = patch ** 2
        for i in range(len(GF)):
            tmp = torch.take(GF[i], index[i])
            norm_tmp = torch.mean(tmp, dim=-1)
            scale_norm = scale * ((norm_tmp - norm_tmp.min()) / (norm_tmp.max() - norm_tmp.min())) + offset
            tmp_bi = torch.as_tensor(scale_norm.repeat_interleave(patch_size)) * 1.0
            GF[i] = GF[i].put_(index[i], tmp_bi)
        return GF
    
def forward(self, data, label, **kwargs):
        """
        The ATT attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        momentum = 0

        delta = self.init_delta(data)
        delta.requires_grad_()
        output = self.model[1](data+delta)
        output.backward(torch.ones_like(output))
        
        resize = transforms.Resize((224, 224))
        GF = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
        GF = resize(GF.reshape(1, 14, 14))
        GF_patchs_t = self.norm_patchs(GF, self.patch_index, self.size, self.scale, self.offset)
        GF_patchs_start = torch.ones_like(GF_patchs_t).cuda() * 0.99
        GF_offset = (GF_patchs_start - GF_patchs_t) / self.epoch
        
        for i in range(self.epoch):
            self.var_A = 0
            self.var_qkv = 0
            self.var_mlp = 0
            self.back_attn = 11
            torch.manual_seed(i)
            random_patch = torch.rand(14, 14).repeat_interleave(16).reshape(14,14*16).repeat(1,16).reshape(224,224).cuda()
            GF_patchs = torch.where(torch.as_tensor(random_patch > GF_patchs_start - GF_offset * (i + 1)), 0., 1.).cuda()
            outputs = self.get_logits(data+delta*GF_patchs.detach())
            loss = self.get_loss(outputs, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        

        return delta.detach()
