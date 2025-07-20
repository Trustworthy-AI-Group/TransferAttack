import torch
from torch.autograd import Function
from torch.nn import Module
from torch import nn

import pywt
import math
import random

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class MetaSSA(MIFGSM):
    """
    MetaSSA Attack
    'Exploring Frequencies via Feature Mixing and Meta-Learning for Improving Adversarial Transferability'(https://arxiv.org//abs/2405.03193)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        n_sample (int): the sample quantity for MetaSSA.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., n_sample=10
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/metassa/resnet50 --attack metassa --model resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/metassa/resnet50 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., n_sample=10, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='MetaSSA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.n_sample = n_sample

    def load_model(self, model_name):
        """
        Override for MetaSSA
        """
        print('loading model for MetaSSA')
        if model_name in models.__dict__.keys():
            print('=> Loading model {} from torchvision.models'.format(model_name))
            model = models.__dict__[model_name](weights="DEFAULT")
        else:
            raise ValueError('Model {} not supported'.format(model_name))
        model = CustomModel(model, model_name)
        return wrap_model(model.eval().cuda())
    
    def craft_adv(self, data, delta, label, feat_x_ll, feat_x_hh, grad_pre):
        gauss = torch.randn(data.size()[0], 3, data.size()[2], data.size()[3]) * self.epsilon * 1
        gauss = gauss.to(self.device)
        x_idct = data + delta + gauss
        LL, LH, HL, HH = DWT(x_idct)
        inputs_hh = IDWT(LL, LH, HL, HH)
        inputs_ll = x_idct - inputs_hh
        outputs = self.model[1](feat_x_ll, feat_x_hh, self.model[0](inputs_ll))
        loss = self.get_loss(outputs, label)
        grad = self.get_grad(loss, delta)

        momentum = self.get_momentum(grad, grad_pre)
        delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta, momentum

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for MetaSSA

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

        # Store clean features of the clean data
        LL, LH, HL, HH = DWT(data)
        inputs_hh_x = IDWT(LL, LH, HL, HH)
        inputs_ll_x = data - inputs_hh_x

        feat_x_ll = self.model[1].featureExtractor(self.model[0](inputs_ll_x))
        feat_x_hh = self.model[1].featureExtractor(self.model[0](inputs_hh_x))

        grad_pre_train = 0
        grad_pre_test = 0

        for _ in range(self.epoch):
            adv_delta = delta.clone()
            adv_train = []
            for n in range(self.n_sample):
                adv_delta, grad_pre_train = self.craft_adv(data, adv_delta, label, feat_x_ll, feat_x_hh, grad_pre_train)
                adv_train.append(adv_delta.clone())

            grad_list_test = []
            for n in range(self.n_sample):
                gauss = torch.randn(data.size()[0], 3, data.size()[2], data.size()[3]) * self.epsilon * 1
                gauss = gauss.to(self.device)
                x_idct = data + adv_train[n] + gauss
                output = self.model[1](feat_x_ll, feat_x_hh, self.model[0](x_idct))
                loss = self.get_loss(output, label)
                grad = self.get_grad(loss, adv_train[n])
                grad_norm = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))
                grad_list_test.append(grad_norm)

            grad = torch.stack(grad_list_test[:self.n_sample]).sum(dim=0)/self.n_sample
            grad_mu = grad + 1 * grad_pre_test
            grad_pre_test = grad_mu

            delta = self.update_delta(delta, data, grad_pre_train+grad_mu, self.alpha)

        return delta.detach()



class DWTFunction_2D_tiny(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        return LL

    @staticmethod
    def backward(ctx, grad_LL):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(grad_LL, matrix_Low_1.t())
        grad_input = torch.matmul(matrix_Low_0.t(), grad_L)
        return grad_input, None, None, None, None


class IDWT_2D_tiny(Module):
    """
    input:  lfc -- (N, C, H/2, W/2)
            hfc_lh -- (N, C, H/2, W/2)
            hfc_hl -- (N, C, H/2, W/2)
            hfc_hh -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """

    def __init__(self, wavename):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \\mathcal{L}, \\mathcal{H}
        :return: self.matrix_low = \\mathcal{L}, self.matrix_high = \\mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                     0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                     0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL):
        """
        recontructing the original 2D data
        the original 2D data = \\mathcal{L}^T * lfc * \\mathcal{L}
                             + \\mathcal{H}^T * hfc_lh * \\mathcal{L}
                             + \\mathcal{L}^T * hfc_hl * \\mathcal{H}
                             + \\mathcal{H}^T * hfc_hh * \\mathcal{H}
        :param LL: the low-frequency component
        :param LH: the high-frequency component, hfc_lh
        :param HL: the high-frequency component, hfc_hl
        :param HH: the high-frequency component, hfc_hh
        :return: the original 2D data
        """
        assert len(LL.size()) == 4
        self.input_height = LL.size()[-2] * 2
        self.input_width = LL.size()[-1] * 2
        import ipdb
        ipdb.set_trace()
        self.get_matrix()
        return IDWTFunction_2D_tiny.apply(LL, self.matrix_low_0, self.matrix_low_1)


class DWT_2D_tiny(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              #hfc_lh: (N, C, H/2, W/2)
              #hfc_hl: (N, C, H/2, W/2)
              #hfc_hh: (N, C, H/2, W/2)
    DWT_2D_tiny only outputs the low-frequency component, which is used in WaveCNet;
    the all four components could be get using DWT_2D, which is used in WaveUNet.
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        # print('band_low', self.band_low, len(self.band_low))  # [1/根号2 = 0.707， 0.07]
        self.band_high = wavelet.rec_hi
        # print('band_high', self.band_high)  # [0.707, -0.707]
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)    # 2
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        print('band_length_half', self.band_length_half)   # 1

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \\mathcal{L}, \\mathcal{H}
        :return: self.matrix_low = \\mathcal{L}, self.matrix_high = \\mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))   # 224
        L = math.floor(L1 / 2)   # 112
        matrix_h = np.zeros((L, L1 + self.band_length - 2))  # (112, 224 + 2 -2)
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        # print('matrix_h_0',  matrix_h_0.shape)
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        # print('matrix_h_1', matrix_h_1.shape)
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                if (index+j) == matrix_g.shape[1]:
                    continue
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                     0:(self.input_height + self.band_length - 2)]
        print('matrix_g_0', matrix_g_0.shape)
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                     0:(self.input_width + self.band_length - 2)]
        print('matrix_g_1', matrix_g_1.shape)

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        print('matrix_h_0', matrix_h_0.shape)
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        print('matrix_h_1', matrix_h_1.shape)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \\mathcal{L} * input * \\mathcal{L}^T
        #input_hfc_lh = \\mathcal{H} * input * \\mathcal{L}^T
        #input_hfc_hl = \\mathcal{L} * input * \\mathcal{H}^T
        #input_hfc_hh = \\mathcal{H} * input * \\mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency component of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0,
                                         self.matrix_high_1)


class IDWTFunction_2D_tiny(Function):
    @staticmethod
    def forward(ctx, input_LL, matrix_Low_0, matrix_Low_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1)
        L = torch.matmul(input_LL, matrix_Low_1.t())
        output = torch.matmul(matrix_Low_0.t(), L)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        return grad_LL, None, None, None, None


class DWT_2D(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \\mathcal{L}, \\mathcal{H}
        :return: self.matrix_low = \\mathcal{L}, self.matrix_high = \\mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                if (index+j) == matrix_g.shape[1]:
                    continue
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                     0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                     0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \\mathcal{L} * input * \\mathcal{L}^T
        input_hfc_lh = \\mathcal{H} * input * \\mathcal{L}^T
        input_hfc_hl = \\mathcal{L} * input * \\mathcal{H}^T
        input_hfc_hh = \\mathcal{H} * input * \\mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class IDWT_2D(Module):
    """
    input:  lfc -- (N, C, H/2, W/2)
            hfc_lh -- (N, C, H/2, W/2)
            hfc_hl -- (N, C, H/2, W/2)
            hfc_hh -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """

    def __init__(self, wavename):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \\mathcal{L}, \\mathcal{H}
        :return: self.matrix_low = \\mathcal{L}, self.matrix_high = \\mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                if (index+j) == matrix_g.shape[1]:
                    continue
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                     0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                     0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        """
        recontructing the original 2D data
        the original 2D data = \\mathcal{L}^T * lfc * \\mathcal{L}
                             + \\mathcal{H}^T * hfc_lh * \\mathcal{L}
                             + \\mathcal{L}^T * hfc_hl * \\mathcal{H}
                             + \\mathcal{H}^T * hfc_hh * \\mathcal{H}
        :param LL: the low-frequency component
        :param LH: the high-frequency component, hfc_lh
        :param HL: the high-frequency component, hfc_hl
        :param HH: the high-frequency component, hfc_hh
        :return: the original 2D data
        """
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0,
                                     self.matrix_high_1)


class IDWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        # L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        L = torch.matmul(input_LH, matrix_High_1.t())
        H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), L), torch.matmul(matrix_High_0.t(), H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        # grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_L = torch.matmul(grad_LH, matrix_High_1.t())
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None
    
DWT = DWT_2D(wavename='haar')
IDWT = IDWT_2D(wavename='haar')

class CustomModel(nn.Module):
    def __init__(self, original_model,model_type):
        super(CustomModel, self).__init__()
        self.features = nn.Sequential()

        # Customize for ResNet
        if 'resnet' in model_type:
            for name, module in original_model.named_children():
                if name not in ['fc']:
                    self.features.add_module(name, module)
        else:
            raise ValueError('The model type is not supported.')
        
        self.fc = original_model.fc

    def forward(self, x1, x2, x3):
        # 随机选择一层
        a = random.uniform(0, 1)
        b = random.uniform(0, 1 - a)
        c = 1 - a - b

        layer_names = list(self.features._modules.keys())
        selected_layer_name = random.choice(layer_names)
        features1 = x1[selected_layer_name].detach()
        features2 = x2[selected_layer_name].detach()
        features3 = self.features[:layer_names.index(selected_layer_name) + 1](x3)

        feat = a * features1 + b * features2 + c * features3

        x = self.features[layer_names.index(selected_layer_name) + 1:](feat)
        x_in = x.view(x.size(0), -1)
        # 继续计算后续层的特征
        x = self.fc(x_in)
        return x

    def featureExtractor(self, x):
        feature_dict = {}
        for name, layer in self.features.named_children():
            x = layer(x)
            # 存储每一层的特征
            feature_dict[name] = x.clone()
        return feature_dict