import torch
import torch.nn.functional as F
import scipy.stats as st

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class POTRIP(MIFGSM):
    """
    Po+Trip Attack
    'Towards Transferable Targeted Attack (CVPR 2020)'(https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Towards_Transferable_Targeted_Attack_CVPR_2020_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the relative size of the resized image
        diversity_prob (float): the probability for transforming the input image
        lamb (float): the weight of triplet loss.
        gamma (float): the margin value.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        kernel_size (int): the size of kernel.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
    
    Official arguments:
        epsilon=16/255, alpha=0.8/255, epoch=20, decay=1., lamb = 0.01, gamma = 0.007, diversity_prob=0.7, kernel_size=5
    
    To reproduce results of the original paper, please change settings as follows:
        in attack.py: 
            model = models.__dict__[model_name](pretrained=True)
        
        in utils.py:
            img_height, img_width = 299, 299
            def load_pretrained_model(cnn_model=[], vit_model=[]):
                for model_name in cnn_model:
                    yield model_name, models.__dict__[model_name](pretrained=True)
    
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/potrip/resnet50_targeted --attack potrip --model=resnet50 --targeted
        python main.py --input_dir ./path/to/data --output_dir adv_data/potrip/resnet50_targeted --eval --targeted
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., 
                lamb = 0.01, gamma = 0.007, diversity_prob=0.7, resize_rate=1.1, targeted=True, random_start=False, kernel_size=5,
                norm='linfty', loss='crossentropy', device=None, attack='POTRIP', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_classes = 1000
        self.lamb =  lamb
        self.gamma = gamma
        self.kernel = self.generate_kernel(kernel_size)
        self.kernel_size = kernel_size
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate
        
    def generate_kernel(self, kernel_size, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)
    
    def Poincare_dis(self, a, b):
        L2_a = torch.sum(torch.square(a), dim=1)
        L2_b = torch.sum(torch.square(b), dim=1)
        diff_sq = torch.sum(torch.square(a - b), dim=1)
        denom = (1 - L2_a) * (1 - L2_b)
        theta = 2 * diff_sq / denom
        distance = torch.mean(torch.acosh(1.0 + theta))
        return distance

    def Cos_dis(self, a, b):
        a_b = torch.abs(torch.sum(torch.multiply(a, b), 1))
        L2_a = torch.sum(torch.square(a), 1)
        L2_b = torch.sum(torch.square(b), 1)
        distance = torch.mean(a_b / (torch.sqrt(L2_a * L2_b) + 1e-8))
        return distance

    def get_loss(self, clean_logits, adv_logits, origi_label, target_label):
        batch_size = clean_logits.shape[0]
        y_tar_onehot = torch.zeros(batch_size, self.num_classes, device=self.device)
        y_tar_onehot.scatter_(1, target_label.unsqueeze(1), 1)  
        y_src_onehot = torch.zeros(batch_size, self.num_classes, device=self.device)
        y_src_onehot.scatter_(1, origi_label.unsqueeze(1), 1)
        u = adv_logits / torch.sum(torch.abs(adv_logits), dim=1, keepdim=True)
        v = torch.clamp((y_tar_onehot - 0.00001), 0.0, 1.0)
        loss_po = self.Poincare_dis(u, v)
        if self.targeted:
            s_tar = self.Cos_dis(y_tar_onehot, adv_logits) 
            s_src = self.Cos_dis(y_src_onehot, adv_logits)             
            loss_trip = torch.clamp(s_src - s_tar + self.gamma, min=0.0, max=2.1)
            loss = -(loss_po + self.lamb * loss_trip) 
            return loss
        else:
            raise NotImplementedError("PoTrip method only supports targeted attacks.")

    def get_grad(self, loss, delta, **kwargs):
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, bias=None, padding=(2,2), groups=3)
        return grad
    
    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return grad + momentum * self.decay 

    def transform(self, x, **kwargs):
        region = int(224*self.resize_rate)
        rnd = np.random.randint(224, region, size=1)[0]
        h_rem = region - rnd
        w_rem = region - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left
        c = np.random.rand(1)
        if c <= self.diversity_prob:
            rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
            padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            return padded
        else:
            return x
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            ori_label = label[0]
            target_label = label[1]
        data = data.clone().detach().to(self.device)
        ori_label = ori_label.clone().detach().to(self.device)
        target_label = target_label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        clean_logits = self.get_logits(self.transform(data))
        momentum = 0
        
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            
            loss = self.get_loss(clean_logits, logits, ori_label, target_label)
            
            grad = self.get_grad(loss, delta)
            
            momentum = self.get_momentum(grad, momentum)
            
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()
