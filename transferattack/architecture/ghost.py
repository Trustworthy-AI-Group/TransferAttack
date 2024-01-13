# example bash: python main.py --attack=ghost_network
from ..utils import *
from ..attack import Attack
from .ghost_networks.resnet import ghost_resnet101, ghost_resnet152
from ..gradient.mifgsm import MIFGSM
from ..gradient.nifgsm import NIFGSM
from ..gradient.vmifgsm import VMIFGSM
from ..input_transformation.dim import DIM
from ..input_transformation.tim import TIM
from ..input_transformation.sim import SIM
from ..input_transformation.admix import Admix



from torch import Tensor
from ..utils import *
from ..gradient.mifgsm import MIFGSM
from ..gradient.nifgsm import NIFGSM
from ..input_transformation.dim import DIM
from ..input_transformation.tim import TIM
from ..input_transformation.sim import SIM
from ..input_transformation.admix import Admix



support_models = {
    "resnet101": ghost_resnet101,
    "resnet152": ghost_resnet152,
}

class GhostNetwork_MIFGSM(MIFGSM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model_name='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model
    
class GhostNetwork_IFGSM(MIFGSM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model_name='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model_name, *args, **kwargs)
        self.decay = 0.

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model

class GhostNetwork_NIFGSM(NIFGSM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model_name='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model

class GhostNetwork_VMIFGSM(VMIFGSM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model

class GhostNetwork_DIM(DIM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model

class GhostNetwork_SIM(SIM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model

class GhostNetwork_TIM(TIM):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """
    def __init__(self, model='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model

class GhostNetwork_Admix(Admix):
    """
    Ghost Network Attack: 

    Arguments:
        model (str): the surrogate model for attack.
        ghost_keep_prob (float): the dropout rate when generating ghost networks.
        ghost_random_range (float): the dropout rate when generating ghost networks of residual structure.
    """

    def __init__(self, model='inc_v3', ghost_keep_prob=0.994, ghost_random_range=0.16, *args, **kwargs):
        self.ghost_keep_prob = ghost_keep_prob          # do not use
        self.ghost_random_range = ghost_random_range    # do not use
        super().__init__(model, *args, **kwargs)

    def load_model(self, model_name):
        if model_name in support_models.keys():
            # The ghost_keep_prob and ghost_random_range are correctly set as param default value,
            # in the __init__ function of each GhostNetwork.
            model = wrap_model(support_models[model_name](weights='DEFAULT').eval().cuda())
        else:
            raise ValueError('Model {} not supported for GhostNetwork'.format(model_name))
        return model