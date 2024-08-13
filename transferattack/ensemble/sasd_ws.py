import torch
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import scipy.stats as st

class SASD_WS(MIFGSM):
    """
    SASD_WS Attack
    'Improving Transferable Targeted Adversarial Attacks with Model Self-Enhancement (CVPR 2024)'(https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Improving_Transferable_Targeted_Adversarial_Attacks_with_Model_Self-Enhancement_CVPR_2024_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        p (float): the probability of weight scaling.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=2.0/255, epoch=300, decay=1., p=0.93

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/sasd_ws/resnet50 --attack sasd_ws --model=resnet50 --targeted
        python main.py --input_dir ./path/to/data --output_dir adv_data/sasd_ws/resnet50 --eval --targeted
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=2.0/255, epoch=300, decay=1., p=0.93, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='SASD_WS', checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.p = p

        
    def load_model(self, model_name, **kwargs):
        weight_path = os.path.join(self.checkpoint_path, 'resnet50_SASD_7.pth')

        if not os.path.exists(weight_path):
            raise ValueError("Please download the checkpoint of the 'resnet50_SASD_Model' from 'https://drive.google.com/drive/folders/1CsNN53GYy9nFcJdSkS5Pcy_faisMDRRh', and put it into the path '{}'.".format(self.checkpoint_path))
        
        model = models.__dict__[model_name](weights="DEFAULT")
        weight = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(weight)
        sasd_model = model.eval().cuda()

        sasd_ws_model = all_scale(sasd_model, p=0.93)

        sasd_ws_model = torch.nn.Sequential(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), sasd_ws_model)

        return sasd_ws_model.eval().cuda()

    ##define TI
    def TI(self, **kwargs):
        def gkern(kernlen=15, nsig=3):
            x = np.linspace(-nsig, nsig, kernlen)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
            return kernel

        kernel_size=5
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to(self.device)
        return gaussian_kernel

    ##define DI
    def DI(self, X_in, **kwargs):
        resize_rate = 1.1
        img_size = X_in.shape[-1]
        img_resize = int(img_size * resize_rate)

        rnd = np.random.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=1)[0]
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = np.random.randint(0, h_rem, size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c <= 0.7:
            X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)), (pad_left, pad_top, pad_right, pad_bottom), mode='constant', value=0)
            return  X_out 
        else:
            return  X_in

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

        gaussian_kernel = self.TI()
        
        momentum = 0
        for _ in range(self.epoch):
            # Calculate the loss
            logits = self.get_logits(self.DI(data+delta))

            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            grad = F.conv2d(grad, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

class AllScaleMethod(prune.BasePruningMethod):
    """
    Scale the parameters in a random way
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, p: float):
        self.p = p
    
    def compute_mask(self, t: torch.Tensor, default_mask):
        tensor_size = t.nelement()

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if tensor_size != 0:
            mask *= self.p

        return mask

    @classmethod
    def apply(cls, module, name, p, importance_scores=None):
        return super(AllScaleMethod, cls).apply(
            module, name, p=p, importance_scores=importance_scores
        )


def AllScaleUnstructured(module, name, p):
    """
    ### Args:
        module: module to prune
        name: parameter name within `module` on which pruning will act.
        amount: decide the amount of parameters to prune
        p: scale the parameters with a ratio of p
    """
    AllScaleMethod.apply(module, name, p)

def all_scale(model, p) -> torch.nn.Module:
    """
    Scale the model
    
    Scale the input model's parameters of convolutional layer in a random way.

    ### Args:
        model: Model to scale.
        scale_rate: Decides the scaling ratio.
        p: Decides the ratio of parameters selected to be applied to the mask.
    
    ### Returns:
        Scaled model.
    """
    list1 = [
        module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
    ]
    parameters_to_prune = list1
    for module in parameters_to_prune:
        AllScaleUnstructured(
            module=module,
            name="weight",
            p=p
        )
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model
