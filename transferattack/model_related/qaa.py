import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

from .qaa_utils.archs.apot import *


class QAA(MIFGSM):
    """
    QAA Attack
    'Quantization Aware Attack: Enhancing Transferable Adversarial Attacks by Model Quantization (TIFS 2024)'(https://arxiv.org/abs/2305.05875)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        quantize_method (str): the quantization method, 'pytorch' or 'apot'.
        w_bit (int): the bit width for weight quantization.
        a_bit (int): the bit width for activation quantization.
        stochastic (bool): whether using stochastic quantization.
        ckpt_id (str): the checkpoint id for apot model.
        ckpt_dir (str): the checkpoint directory for apot model.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., quantize_method='apot', w_bit=2, a_bit=2, stochastic=True, ckpt_id='120603', ckpt_dir='/path/to/checkpoints'

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/qaa/resnet50 --attack qaa --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/qaa/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=5, num_admix=3, admix_strength=0.2, 
                targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='QAA', quantize_method='apot', 
                w_bit=2, a_bit=2, stochastic=True, ckpt_id='120603', ckpt_dir='path/to/checkpoints', **kwargs):
        self.quantize_method = quantize_method
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.stochastic = stochastic
        self.ckpt_id = ckpt_id
        self.ckpt_dir = ckpt_dir
        
        super().__init__(model_name, epsilon, alpha, epoch, decay,
                         targeted, random_start, norm, loss, device, attack)

    def load_model(self, model_name):
        if self.quantize_method == "pytorch":
            assert self.w_bit == 8 and self.a_bit == 8
            import torchvision.models.quantization as models
            model = models.__dict__[model_name](pretrained=True, quantize=True)
            print("8-bit model {} loaded successfully!".format(model_name))
        elif self.quantize_method == "apot":
            if self.stochastic == True and self.ckpt_id == '120603':
                from .qaa_utils.archs import apot as models
                model = models.__dict__[model_name](
                    pretrained=False, bit=self.w_bit, stochastic=True)
                model = torch.nn.DataParallel(model).cuda()
                model_dir = os.path.join(self.ckpt_dir, "apot", model_name +
                                         "_w{}a{}_stochastic_120603.pth.tar".format(self.w_bit, self.a_bit))
                model.load_state_dict(torch.load(model_dir)["state_dict"])
            else:
                from .qaa_utils.archs import apot as models
                model = models.__dict__[model_name](
                    pretrained=False, bit=self.w_bit, stochastic=False)
                model = torch.nn.DataParallel(model).cuda()
                model_dir = os.path.join(
                    self.ckpt_dir, "apot", model_name + "_w{}a{}.pth.tar".format(self.w_bit, self.a_bit))
                model.load_state_dict(torch.load(model_dir)["state_dict"])
            print("model successfully loaded from {}".format(model_dir))
        else:
            raise Exception(
                'quantize method {} not implemented!'.format(self.quantize_method))

        return model
