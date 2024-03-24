from typing import Any
import math
import torch
import random
from PIL import ImageOps
from ..utils import *
from ..attack import Attack
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Dropout
import copy
import pdb

softmax = torch.nn.Softmax(dim=0)
def select_op(op_params, num_ops):
    prob = softmax(op_params)
    op_ids = torch.multinomial(prob, num_ops, replacement=True).tolist()
    return op_ids

def trace_prob(op_params, op_ids):
    probs = softmax(op_params)
    tp = 1
    for idx in op_ids:
        tp = tp * probs[idx]
    return tp


class RWAug_Search: 
    def __init__(self, n, idxs):
        self.n = n
        #idxs is the operation id
        self.idxs = idxs      
        self.op_list = op_list

    def __call__(self, img):
      assert len(self.idxs) == self.n
      #print(self.idxs)
      for idx in self.idxs:
        img = op_list[idx](img)
      return img


def vertical_shift(x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low = 0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)

def vertical_flip(x):
    return x.flip(dims=(2,))

def horizontal_flip(x):
    return x.flip(dims=(3,))

def rotate45(x):
    return transforms.functional.rotate(img=x, angle=45)

def rotate135(x):
    return transforms.functional.rotate(img=x, angle=135)

def rotate90(x):
    return x.rot90(k=1, dims=(2,3))

def rotate180(x):
    return x.rot90(k=2, dims=(2,3))


def add_noise(x):
    return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)

def identity(x):
    return x

class rotate():
    def __init__(self, angle, num_scale) -> None:
        self.num_scale = num_scale
        self.angle = angle
    
    def __call__(self, x):
        return torch.cat([transforms.functional.rotate(img=x, angle=(self.angle / (2**i))) for i in range(self.num_scale)])


class sim():
    def __init__(self, num_copy) -> None:
        self.num_copy = num_copy
    
    def __call__(self, x):
        return torch.cat([x / (2**i) for i in range(self.num_copy)])
        #return torch.cat([x / (2**self.num_copy)])

class dim():
    def __init__(self, resize_rate=1.1, diversity_prob=0.5) -> None:
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        
    def __call__(self, x):
        """
        Random transform the input images
        """
        # do not transform the input image
        #if torch.rand(1) > self.diversity_prob:
        #    return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)


class blockshuffle():
    def __init__(self, num_block=3, num_scale=10) -> None:
        self.num_block = num_block
        self.num_scale = num_scale
        
    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        # perm = torch.randperm(self.num_block)
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def __call__(self, x, **kwargs):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])

class admix():
    def __init__(self, num_admix=3, admix_strength=0.2, num_scale=3) -> None:
        self.num_scale = num_scale
        self.num_admix = num_admix
        self.admix_strength = admix_strength
        
    def __call__(self, x) -> Any:
        admix_images = torch.concat([(x + self.admix_strength * x[torch.randperm(x.size(0))].detach()) for _ in range(self.num_admix)], dim=0)
        return torch.concat([admix_images / (2 ** i) for i in range(self.num_scale)])
    
class ide():
    def __init__(self, dropout_prob=[0,0.1,0.2,0.3,0.4,0.5]) -> None:
        self.dropout_prob = dropout_prob
        
    def __call__(self, x):
        return torch.cat([Dropout(p=prob)(x)*(1-prob) for prob in self.dropout_prob])
    
'''class masked():
    def __init__(self, patch_size) -> None:
        self.patch_size = patch_size
        self.num = 0
    
    def __call__(self, x):
        _, _, w, h = x.shape
        y_axis = [i for i in range(0, h+1, self.patch_size)]
        x_axis = [i for i in range(0, w+1, self.patch_size)]
        self.num = 0
        xs = []
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                x_copy = x.clone()
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = 0
                xs.append(x_copy)
                self.num += 1
        return torch.cat(xs, dim=0)'''


class masked():
    def __init__(self, num_block, num_scale=5) -> None:
        self.num_block = num_block
        self.num_scale = num_scale
        
    def blockmask(self, x, choice=-1):
        _, _, w, h = x.shape
        
        if w == h:
            step = w / self.num_block
            points = [round(step * i) for i in range(self.num_block + 1)]
        
        x_copy = x.clone()
        x_block, y_block = random.randint(0, self.num_block-1), random.randint(0, self.num_block-1)
        x_copy[:, :, points[x_block]:points[x_block+1], points[y_block]:points[y_block+1]] = 0
        
        return x_copy
    
    def __call__(self, x):
        return torch.cat([self.blockmask(x) for _ in range(self.num_scale)])
    
class ssm():
    def __init__(self, rho=0.5, num_spectrum=10):
        self.epsilon = 16/255
        self.rho = rho
        self.num_spectrum = num_spectrum
    
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
    
    def __call__(self, x):
        x_idct = []
        
        for _ in range(self.num_spectrum):
            gauss = torch.randn(x.size()[0], 3, 224, 224) * self.epsilon
            gauss = gauss.cuda()
            x_dct = self.dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
            x_idct.append(self.idct_2d(x_dct * mask))

        return torch.cat(x_idct)

class crop():
    def __init__(self, ratio,  num_scale=5) -> None:
        self.num_scale = num_scale
        self.ratio = ratio
        
    def crop(self, x, ratio):
        width = int(x.shape[2]*ratio)
        height = int(x.shape[3]*ratio)
        
        left = 0+(x.shape[2]-width)//2
        top = 0+(x.shape[3]-height)//2
        return transforms.functional.resized_crop(x, top, left, height, width, (224, 224))
        
    def __call__(self, x) -> Any:
        #transforms.functional.resized_crop(x, 0, 0, int(0.9*224), int(0.9*224), (224, 224))
        return torch.cat([self.crop(x, self.ratio+(1-self.ratio)*(i+1)/self.num_scale) for i in range(self.num_scale)])

class affine():
    def __init__(self, offset, num_scale=5) -> None:
        self.num_scale = num_scale
        self.offset = offset
        
    def __call__(self, x):
        return torch.cat([transforms.functional.affine(img=x, angle=0, translate=[self.offset*(i+1)/self.num_scale, self.offset*(i+1)/self.num_scale], scale=1, shear=0) for i in range(self.num_scale)])
        
        
        
op_list = [identity, #0
           rotate(30,5), rotate(60,5), rotate(90,5), rotate(120,5), rotate(150,5),rotate(180,5),rotate(210,5),rotate(240,5),rotate(270,5),rotate(300,5), #1-10
           sim(1), sim(2), sim(3), sim(4), sim(5),sim(6),sim(7),sim(8),sim(9),sim(10), #11-20
           dim(1.1),dim(1.15),dim(1.2),dim(1.25),dim(1.3),dim(1.35),dim(1.4),dim(1.45),dim(1.5),dim(1.55), #21-30
           blockshuffle(3), blockshuffle(4), blockshuffle(5), blockshuffle(6), blockshuffle(7),blockshuffle(8),blockshuffle(9),blockshuffle(10),blockshuffle(11),blockshuffle(12), #31-40
           admix(1,0.2),admix(2,0.2),admix(3,0.2),admix(4,0.2),admix(5,0.2),admix(1,0.4),admix(2,0.4),admix(3,0.4),admix(4,0.4),admix(5,0.4), #41-50
           ide([0.1]), ide([0.1,0.2]), ide([0.1,0.2,0.3]), ide([0.1,0.2,0.3,0.4]), ide([0.1,0.2,0.3,0.4,0.5]),ide([0.2,0.3,0.4,0.5]), ide([0.1,0.3,0.4,0.5]), ide([0.1,0.2,0.4,0.5]), ide([0.1,0.2,0.3,0.5]), ide([0.1,0.2,0.3,0.4]), #51-60
           masked(2), masked(4), masked(6), masked(8), masked(10),masked(3), masked(5), masked(7), masked(9), masked(11), # 61-70
           ssm(0.2), ssm(0.4), ssm(0.5), ssm(0.6), ssm(0.8), ssm(0.1), ssm(0.3), ssm(0.7), ssm(0.9), # 71-80
           crop(0.1), crop(0.2), crop(0.3), crop(0.4), crop(0.5), crop(0.6), crop(0.7), crop(0.8), crop(0.9), # 81-90
           affine(0.5), affine(0.55), affine(0.6), affine(0.65), affine(0.7), affine(0.75), affine(0.8), affine(0.85), affine(0.9), # 91-100
          ] 


#op_list = [vertical_shift, horizontal_shift, vertical_flip, horizontal_flip, rotate180, scale, add_noise]


class L2T(Attack):
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='MI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_scale = kwargs['num_scale']
        

    def get_loss(self, logits, label, num_copy):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        return - self.loss(logits, label.repeat(num_copy)) if self.targeted else self.loss(logits, label.repeat(num_copy))

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def transform(self, x, **kwargs):
        return kwargs['search'](x)
    
    '''def load_model(self, model_name):
        """
        The model Loading stage, which should be overridden when surrogate model is customized (e.g., DSM, SE_TR, etc.)
        Prioritize the model in torchvision.models, then timm.models

        Arguments:
            model_name (str): the name of surrogate model in model_list in utils.py

        Returns:
            model (torch.nn.Module): the surrogate model wrapped by wrap_model in utils.py
        """
        model_list = []
        for name in model_name.split('_'):
            if name in models.__dict__.keys():
                print('=> Loading model {} from torchvision.models'.format(name))
                model = models.__dict__[name](weights="IMAGENET1K_V1")
            elif name in timm.list_models():
                print('=> Loading model {} from timm.models'.format(name))
                model = timm.create_model(name, pretrained=True)
            else:
                raise ValueError('Model {} not supported'.format(name))
            model_list.append(model.eval().cuda())
        return EnsembleModel(model_list)'''

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

            aug_length = len(op_list)
            ops_num = 2
            learning_rate = 0.01
            #self.num_scale = 10

            aug_param = torch.nn.Parameter(torch.zeros(aug_length,requires_grad=True),requires_grad=True)      

            data = data.clone().detach().to(self.device)
            label = label.clone().detach().to(self.device)

            # Initialize adversarial perturbation
            delta = self.init_delta(data)

            momentum = 0
            for e in range(self.epoch):
                # transform data
                aug_probs = []
                losses = []
                
                for i in range(self.num_scale):
                    rw_search = RWAug_Search(ops_num, [0,0])
                    
                    augtype = (ops_num, select_op(aug_param, ops_num))
                    aug_prob = trace_prob(aug_param, augtype[1])

                    rw_search.n = augtype[0]
                    rw_search.idxs = augtype[1]
                    
                    aug_probs.append(aug_prob)
                    
                    logits = self.get_logits(self.transform(data+delta, search=rw_search))
                    
                    losses.append(self.get_loss(logits, label, math.floor((len(logits)+0.01)/len(label))).reshape(1))

                # Calculate the loss
                loss = torch.sum(torch.cat(losses))/self.num_scale
                
                # Calculate the gradients
                grad = self.get_grad(loss, delta)
                
                aug_losses = torch.cat([aug_probs[i]*losses[i].reshape(1) for i in range(self.num_scale)])
                aug_loss = torch.sum(aug_losses)/self.num_scale
                
                aug_grad = torch.autograd.grad(aug_loss, aug_param, retain_graph=False, create_graph=False)[0]
                aug_param = aug_param + learning_rate * aug_grad

                # Calculate the momentum
                momentum = self.get_momentum(grad, momentum)

                # Update adversarial perturbation
                delta = self.update_delta(delta, data, momentum, self.alpha)   
            #print(softmax(aug_param))
            #print(aug_param)
            return delta.detach()