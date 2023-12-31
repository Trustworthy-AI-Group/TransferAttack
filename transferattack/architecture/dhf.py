# example bash: python main.py --attack=mifgsm_dhf
from torch import Tensor
from ..utils import *
from .dhf_networks.inception import dhf_inception_v3
from .dhf_networks.inc_res_v2 import dhf_inc_res_v2
from .dhf_networks.resnet import dhf_resnet18, dhf_resnet50, dhf_resnet101, dhf_resnet152
from ..gradient.mifgsm import MIFGSM
from ..gradient.nifgsm import NIFGSM
from ..input_transformation.dim import DIM
from ..input_transformation.tim import TIM
from ..input_transformation.sim import SIM
from ..input_transformation.admix import Admix
from .dhf_networks import utils


support_models = {
    "inc_v3":  dhf_inception_v3,
    "inc_res": dhf_inc_res_v2,
    'resnet18': dhf_resnet18,
    "resnet50": dhf_resnet50,
    "resnet101": dhf_resnet101,
    "resnet152": dhf_resnet152,
}

"""
Diversifying the High-level Features for better Adversarial Transferability BMVC 2023 (https://arxiv.org/abs/2304.10136)
"""

class DHF_IFGSM(MIFGSM):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)
        self.decay = 0.

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = data.clone().detach().to(self.device).requires_grad_(False)
        self.update_mixup_feature(self.benign_images)
        # return super().forward(data, label, **kwargs)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def preprocess(self, *args, **kwargs):
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])


class DHF_MIFGSM(MIFGSM):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def preprocess(self, *args, **kwargs):
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = data.clone().detach().to(self.device).requires_grad_(False)
        self.update_mixup_feature(self.benign_images)
        # return super().forward(data, label, **kwargs)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()


class DHF_NIFGSM(NIFGSM):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = data.clone().detach().to(self.device).requires_grad_(False)
        self.update_mixup_feature(self.benign_images)
        # return super().forward(data, label, **kwargs)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def preprocess(self, *args, **kwargs):
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])


class DHF_DIM(DIM):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = data.clone().detach().to(self.device).requires_grad_(False)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def preprocess(self, *args, **kwargs):
        self.reuse_rnds = False
        mixup_input = self.transform(self.benign_images)
        self.update_mixup_feature(mixup_input)
        self.reuse_rnds = True
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])


class DHF_TIM(TIM):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = data.clone().detach().to(self.device).requires_grad_(False)
        self.update_mixup_feature(self.benign_images)
        # return super().forward(data, label, **kwargs)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def preprocess(self, *args, **kwargs):
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])


class DHF_SIM(SIM):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = self.transform(data.clone().detach().to(self.device).requires_grad_(False))
        self.update_mixup_feature(self.benign_images)
        # return super().forward(data, label, **kwargs)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def preprocess(self, *args, **kwargs):
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])    


class DHF_Admix(Admix):
    """
    DHF Attack

    Arguments:
        model (str): the surrogate model name for attack.
        mixup_weight_max (float): the maximium of mixup weight.
        random_keep_prob (float): the keep probability when adjusting the feature elements.
    """

    def __init__(self, model_name='inc_v3', dhf_modules=None, mixup_weight_max=0.2, random_keep_prob=0.9, *args, **kwargs):
        self.dhf_moduels = dhf_modules
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        self.benign_images = None
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            model = wrap_model(support_models[model_name](mixup_weight_max=self.mixup_weight_max, 
                random_keep_prob=self.random_keep_prob, weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for DHF'.format(model_name))
        return model

    def update_mixup_feature(self, data: Tensor):
        utils.turn_on_dhf_update_mf_setting(model=self.model)
        _ = self.model(data)
        utils.trun_off_dhf_update_mf_setting(model=self.model)

    def forward(self, data: Tensor, label: Tensor, **kwargs):
        self.benign_images = data.clone().detach().to(self.device).requires_grad_(False)
        # self.update_mixup_feature(self.benign_images)
        # return super().forward(data, label, **kwargs)
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        # Initialize correct indicator 
        num_scale = 1 if not hasattr(self, "num_scale") else self.num_scale
        num_scale = num_scale if not hasattr(self, "num_admix") else num_scale * self.num_admix
        correct_indicator = torch.ones(size=(len(data)*num_scale,), device=self.device)

        momentum = 0
        for _ in range(self.epoch):
            self.preprocess(correct_indicator=correct_indicator)
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Update correct indicator
            correct_indicator = (torch.max(logits.detach(), dim=1)[1] == label.repeat(num_scale)).to(torch.float32)
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def preprocess(self, *args, **kwargs):
        self.reuse_indices = False
        mixup_input = self.transform(self.benign_images)
        self.update_mixup_feature(mixup_input)
        self.reuse_indices = True
        utils.turn_on_dhf_attack_setting(self.model, dhf_indicator=1-kwargs["correct_indicator"])