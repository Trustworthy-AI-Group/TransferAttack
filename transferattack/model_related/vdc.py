import numpy as np
import torch
import torch.nn as nn
from functools import partial
import random

from ..gradient.mifgsm import MIFGSM
from ..utils import *

class VDC(MIFGSM):
    """
    VDC(Virtual Dense Connection) Attack
    'Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection (AAAI 2024)'(https://ojs.aaai.org/index.php/AAAI/article/view/28541)

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
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/vdc/vit --attack vdc --model=vit_base_patch16_224
        python main.py --input_dir ./path/to/data --output_dir adv_data/vdc/vit --eval
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,  targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='VDC', sample_num_batches=130, lamb=0.1):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.model_name = model_name
        self.lamb = lamb
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        self.record_grad = []
        self.record_grad_mlp = []
        ###############
        self.attn_record = []
        self.mlp_record = []
        self.attn_add = []
        self.mlp_add = []
        self.norm_list = []
        self.stage = []
        self.attn_block = 0
        self.mlp_block = 0
        self.hooks = []
        
        self.skip_record = []
        self.skip_add = []
        self.skip_block = 0
        
        assert self.sample_num_batches <= self.max_num_batches
    
    def _register_model(self, add = False): 
        ####################################
        #vit
        def mlp_record_vit_stage(module, grad_in, grad_out, gamma):
            #print("record", self.mlp_block)
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            #ablation
            grad_record = grad_in[0].data.cpu().numpy() * 0.1*(0.5**(self.mlp_block))
            #grad_record = grad_in[0].data.cpu().numpy()
            if self.mlp_block == 0:
                grad_add = np.zeros_like(grad_record)
                #ablation
                grad_add[:,0,:] = self.norm_list[:,0,:]* 0.1*(0.5)
                #grad_add[:,0,:] = self.norm[:,0,:]
                self.mlp_add.append(grad_add)
                self.mlp_record.append(grad_record+grad_add) 
            else:
                self.mlp_add.append(self.mlp_record[-1])
                total_mlp = self.mlp_record[-1] + grad_record
                self.mlp_record.append(total_mlp) 
            self.mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def mlp_add_vit(module, grad_in, grad_out, gamma):
            #print("add", self.mlp_block)
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            ####
            #mask_0 = torch.zeros_like(grad_in[0])
            ####
            out_grad = mask * grad_in[0][:]
            #out_grad = torch.where(grad_in[0][:] > 0, mask * grad_in[0][:], mask_0 * grad_in[0][:])
            out_grad += torch.tensor(self.mlp_add[self.mlp_block]).cuda()
            self.mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])
        
        def attn_record_vit_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            #ablation
            grad_record = grad_in[0].data.cpu().numpy() * 0.1*(0.5**(self.attn_block))
            #grad_record = grad_in[0].data.cpu().numpy()
            if self.attn_block == 0:
                self.attn_add.append(np.zeros_like(grad_record))
                self.attn_record.append(grad_record) 
            else:
                self.attn_add.append(self.attn_record[-1])
                total_attn = self.attn_record[-1] + grad_record
                self.attn_record.append(total_attn) 
            
            self.attn_block += 1
            return (out_grad, )

        def attn_add_vit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            ####
            #mask_0 = torch.zeros_like(grad_in[0])
            ####
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            #out_grad = torch.where(grad_in[0][:] > 0, mask * grad_in[0][:], mask_0 * grad_in[0][:])
            out_grad += torch.tensor(self.attn_add[self.attn_block]).cuda()
            self.attn_block += 1
            return (out_grad, )
            
        def norm_record_vit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in  
        
        ####################################################
        # pit
        def pool_record_pit(module, grad_in, grad_out, gamma):
            grad_add = grad_in[0].data
            B,C,H,W = grad_add.shape
            grad_add = grad_add.reshape((B,C,H*W)).transpose(1,2)
            self.stage.append(grad_add.cpu().numpy())
            return grad_in
        
        def mlp_record_pit_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.mlp_block < 4:
                grad_record = grad_in[0].data.cpu().numpy() * 0.03*(0.5**(self.mlp_block))
                if self.mlp_block == 0:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:,0,:] = self.norm_list[:,0,:]* 0.03*(0.5)
                    self.mlp_add.append(grad_add)
                    self.mlp_record.append(grad_record+grad_add) 
                else:
                    self.mlp_add.append(self.mlp_record[-1])
                    total_mlp = self.mlp_record[-1] + grad_record
                    self.mlp_record.append(total_mlp) 
            elif self.mlp_block < 10:
                grad_record = grad_in[0].data.cpu().numpy() * 0.03*(0.5**(self.mlp_block))
                if self.mlp_block == 4:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:,1:,:] = self.stage[0]* 0.03*(0.5)
                    self.mlp_add.append(grad_add)
                    self.mlp_record.append(grad_record+grad_add) 
                else:
                    self.mlp_add.append(self.mlp_record[-1])
                    #total_mlp = self.mlp_record[-1] + grad_record
                    total_mlp = self.mlp_record[-1]
                    self.mlp_record.append(total_mlp)
            else:
                grad_record = grad_in[0].data.cpu().numpy() * 0.03*(0.5**(self.mlp_block))
                if self.mlp_block == 10:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:,1:,:] = self.stage[1]* 0.03*(0.5)
                    self.mlp_add.append(grad_add)
                    self.mlp_record.append(grad_record+grad_add) 
                else:
                    self.mlp_add.append(self.mlp_record[-1])
                    #total_mlp = self.mlp_record[-1] + grad_record
                    total_mlp = self.mlp_record[-1]
                    self.mlp_record.append(total_mlp)
            self.mlp_block += 1
            
            return (out_grad, grad_in[1], grad_in[2])
            
        def mlp_add_pit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(self.mlp_add[self.mlp_block]).cuda()
            self.mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])
            
        def attn_record_pit_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.attn_block < 4:
                grad_record = grad_in[0].data.cpu().numpy() * 0.03*(0.5**(self.attn_block))
                if self.attn_block == 0:
                    self.attn_add.append(np.zeros_like(grad_record))
                    self.attn_record.append(grad_record) 
                else:
                    self.attn_add.append(self.attn_record[-1])
                    total_attn = self.attn_record[-1] + grad_record
                    self.attn_record.append(total_attn) 
            elif self.attn_block < 10:
                grad_record = grad_in[0].data.cpu().numpy() * 0.03*(0.5**(self.attn_block))
                if self.attn_block == 4:
                    self.attn_add.append(np.zeros_like(grad_record))
                    self.attn_record.append(grad_record) 
                else:
                    self.attn_add.append(self.attn_record[-1])
                    #total_attn = self.attn_record[-1] + grad_record
                    total_attn = self.attn_record[-1]
                    self.attn_record.append(total_attn)
            else:
                grad_record = grad_in[0].data.cpu().numpy() * 0.03*(0.5**(self.attn_block))
                if self.attn_block == 10:
                    self.attn_add.append(np.zeros_like(grad_record))
                    self.attn_record.append(grad_record) 
                else:
                    self.attn_add.append(self.attn_record[-1])
                    #total_attn = self.attn_record[-1] + grad_record
                    total_attn = self.attn_record[-1]
                    self.attn_record.append(total_attn)
            self.attn_block += 1
            return (out_grad, )
            
        def attn_add_pit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(self.attn_add[self.attn_block]).cuda()
            self.attn_block += 1
            return (out_grad, )
            
        def norm_record_pit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in   
               
        ####################################################
        # visformer
        def pool_record_vis(module, grad_in, grad_out, gamma):
            grad_add = grad_in[0].data
            #B,C,H,W = grad_add.shape
            #grad_add = grad_add.reshape((B,C,H*W)).transpose(1,2)
            self.stage.append(grad_add.cpu().numpy())
            return grad_in
        
        def mlp_record_vis_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.mlp_block < 4:
                grad_record = grad_in[0].data.cpu().numpy() * 0.1*(0.5**(self.mlp_block))
                if self.mlp_block == 0:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:,0,:] = self.norm_list[:,0,:]* 0.1*(0.5)
                    self.mlp_add.append(grad_add)
                    self.mlp_record.append(grad_record+grad_add) 
                else:
                    self.mlp_add.append(self.mlp_record[-1])
                    total_mlp = self.mlp_record[-1] + grad_record
                    self.mlp_record.append(total_mlp) 
            else:
                grad_record = grad_in[0].data.cpu().numpy() * 0.1*(0.5**(self.mlp_block))
                if self.mlp_block == 4:
                    grad_add = np.zeros_like(grad_record)
                    #grad_add[:,1:,:] = self.stage[0]* 0.1*(0.5)
                    self.mlp_add.append(grad_add)
                    self.mlp_record.append(grad_record+grad_add) 
                else:
                    self.mlp_add.append(self.mlp_record[-1])
                    total_mlp = self.mlp_record[-1] + grad_record
                    self.mlp_record.append(total_mlp)

            self.mlp_block += 1
            
            return (out_grad, grad_in[1], grad_in[2])
            
        def mlp_add_vis(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(self.mlp_add[self.mlp_block]).cuda()
            self.mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])
            
        def norm_record_vis(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in

        def attn_record_vis_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.attn_block < 4:
                grad_record = grad_in[0].data.cpu().numpy() * 0.1*(0.5**(self.attn_block))
                if self.attn_block == 0:
                    self.attn_add.append(np.zeros_like(grad_record))
                    self.attn_record.append(grad_record) 
                else:
                    self.attn_add.append(self.attn_record[-1])
                    total_attn = self.attn_record[-1] + grad_record
                    self.attn_record.append(total_attn) 
            else:
                grad_record = grad_in[0].data.cpu().numpy() * 0.1*(0.5**(self.attn_block))
                if self.attn_block == 4:
                    self.attn_add.append(np.zeros_like(grad_record))
                    self.attn_record.append(grad_record) 
                else:
                    self.attn_add.append(self.attn_record[-1])
                    total_attn = self.attn_record[-1] + grad_record
                    self.attn_record.append(total_attn)
            
            self.attn_block += 1
            return (out_grad, )
            
        def attn_add_vis(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(self.attn_add[self.attn_block]).cuda()
            self.attn_block += 1
            return (out_grad, )
          
        ###########
        # vit
        mlp_record_func_vit = partial(mlp_record_vit_stage, gamma=1.0)
        norm_record_func_vit = partial(norm_record_vit, gamma=1.0)
        mlp_add_func_vit = partial(mlp_add_vit, gamma=0.5)
        attn_record_func_vit = partial(attn_record_vit_stage, gamma=1.0)
        attn_add_func_vit = partial(attn_add_vit, gamma=0.25)
        ###########
        # pit
        attn_record_func_pit = partial(attn_record_pit_stage, gamma=1.0)
        mlp_record_func_pit = partial(mlp_record_pit_stage, gamma=1.0)
        norm_record_func_pit = partial(norm_record_pit, gamma=1.0)
        pool_record_func_pit = partial(pool_record_pit, gamma=1.0)
        attn_add_func_pit = partial(attn_add_pit, gamma=0.25)
        mlp_add_func_pit = partial(mlp_add_pit, gamma=0.5)
        #mlp_add_func_pit = partial(mlp_add_pit, gamma=0.75)
        
        ###########
        # visformer
        attn_record_func_vis = partial(attn_record_vis_stage, gamma=1.0)
        mlp_record_func_vis = partial(mlp_record_vis_stage, gamma=1.0)
        norm_record_func_vis = partial(norm_record_vis, gamma=1.0)
        pool_record_func_vis = partial(pool_record_vis, gamma=1.0)
        attn_add_func_vis = partial(attn_add_vis, gamma=0.25)
        mlp_add_func_vis = partial(mlp_add_vis, gamma=0.5)
        
        
        if add == False:
            if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                hook = self.model[1].norm.register_backward_hook(norm_record_func_vit)
                self.hooks.append(hook)
                for i in range(12):
                    hook = self.model[1].blocks[i].norm2.register_backward_hook(mlp_record_func_vit)
                    self.hooks.append(hook)
                    hook = self.model[1].blocks[i].attn.attn_drop.register_backward_hook(attn_record_func_vit)
                    self.hooks.append(hook)
            elif self.model_name == 'pit_b_224':
                hook = self.model[1].norm.register_backward_hook(norm_record_func_pit)
                self.hooks.append(hook)
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
                    hook = self.model[1].transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_record_func_pit)
                    self.hooks.append(hook)
                    #hook = self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_record_func_pit)
                    hook = self.model[1].transformers[transformer_ind].blocks[used_block_ind].norm2.register_backward_hook(mlp_record_func_pit)
                    self.hooks.append(hook)
                hook = self.model[1].transformers[0].pool.register_backward_hook(pool_record_func_pit)
                self.hooks.append(hook)
                hook = self.model[1].transformers[1].pool.register_backward_hook(pool_record_func_pit)
                self.hooks.append(hook)
            elif self.model_name == 'visformer_small':
                hook = self.model[1].norm.register_backward_hook(norm_record_func_vis)
                self.hooks.append(hook)
                for block_ind in range(8):
                    if block_ind < 4:
                        hook = self.model[1].stage2[block_ind].attn.attn_drop.register_backward_hook(attn_record_func_vis)
                        self.hooks.append(hook)
                        #hook = self.model.stage2[block_ind].mlp.register_backward_hook(mlp_record_func_vis)
                        hook = self.model[1].stage2[block_ind].norm2.register_backward_hook(mlp_record_func_vis)
                        self.hooks.append(hook)
                    elif block_ind >=4:
                        hook = self.model[1].stage3[block_ind-4].attn.attn_drop.register_backward_hook(attn_record_func_vis)
                        self.hooks.append(hook)
                        #hook = self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_record_func_vis)
                        hook = self.model[1].stage3[block_ind-4].norm2.register_backward_hook(mlp_record_func_vis)
                        self.hooks.append(hook)
                hook = self.model[1].patch_embed3.register_backward_hook(pool_record_func_vis)
                self.hooks.append(hook)
                hook = self.model[1].patch_embed2.register_backward_hook(pool_record_func_vis)
                self.hooks.append(hook)
        elif add == True:
            if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                for i in range(12):
                    hook = self.model[1].blocks[i].norm2.register_backward_hook(mlp_add_func_vit)
                    self.hooks.append(hook)
                    hook = self.model[1].blocks[i].attn.attn_drop.register_backward_hook(attn_add_func_vit)
                    self.hooks.append(hook)
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
                    hook = self.model[1].transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_add_func_pit)
                    self.hooks.append(hook)
                    #hook = self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_add_func_pit)
                    hook = self.model[1].transformers[transformer_ind].blocks[used_block_ind].norm2.register_backward_hook(mlp_add_func_pit)
                    self.hooks.append(hook)
            elif self.model_name == 'visformer_small':
                for block_ind in range(8):
                    if block_ind < 4:
                        hook = self.model[1].stage2[block_ind].attn.attn_drop.register_backward_hook(attn_add_func_vis)
                        self.hooks.append(hook)
                        #hook = self.model.stage2[block_ind].mlp.register_backward_hook(mlp_add_func_vis)
                        hook = self.model[1].stage2[block_ind].norm2.register_backward_hook(mlp_add_func_vis)
                        self.hooks.append(hook)
                    elif block_ind >=4:
                        hook = self.model[1].stage3[block_ind-4].attn.attn_drop.register_backward_hook(attn_add_func_vis)
                        self.hooks.append(hook)
                        #hook = self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_add_func_vis)
                        hook = self.model[1].stage3[block_ind-4].norm2.register_backward_hook(mlp_add_func_vis)
                        self.hooks.append(hook)
    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

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
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation
    
    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def forward(self, data, label, **kwargs):
        """
        The VDC attack procedure

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

        for i in range(self.epoch):
            self.attn_record = []
            self.attn_add = []
            self.mlp_record = []
            self.mlp_add = []
            self.skip_record = []
            self.skip_add = []
            self.mlp_block = 0
            self.attn_block = 0
            self.skip_block = 0
            self._register_model(add = False)
            
            outputs = self.model(data + delta)
            cost = self.get_loss(outputs, label)
            cost.backward()
            delta.grad.data.zero_()
            for hook in self.hooks:
                hook.remove()
            
            self.mlp_block = 0
            self.attn_block = 0
            self.skip_block = 0
            self._register_model(add = True)
            outputs = self.model(data + delta)
            cost = self.get_loss(outputs, label)
            cost.backward()
            grad = delta.grad.data

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # delta.grad.data.zero_()
            delta = self.update_delta(delta, data, momentum, self.alpha)
            
            for hook in self.hooks:
                hook.remove()
        return delta.detach()