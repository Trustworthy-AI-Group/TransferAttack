import torch
from torch import nn, Tensor
# from torchview import draw_graph


class DHFUnit(nn.Module):
    def __init__(self, mixup_weight_max=0.2, random_keep_prob=0.9) -> None:
        super().__init__()
        # dhf static params
        self.mixup_weight_max = mixup_weight_max
        self.random_keep_prob = random_keep_prob
        # dhf dynamic params
        self.if_dhf = False
        self.update_mf = False
        self.dhf_indicator= None
        # dhf self-update param
        self.mixup_feature = None

    def set_dhf_params(self, if_dhf: bool, update_mf: bool, dhf_indicator: Tensor):
        self.if_dhf = if_dhf
        self.update_mf = update_mf
        self.dhf_indicator= dhf_indicator

    def forward(self, x: Tensor):
        if self.if_dhf:
            x = self._forward(x)
        if self.update_mf:
            self.mixup_feature = x.detach().clone().requires_grad_(False)
        return x

    def _forward(self, x):
        '''
            x: input tensor, shape (b, c, w, h).
            dhf_indicator: input tensor, shape(b,), indicating which image is applied to dhf.
            mixup_feature: mixup tensor, shape (b, c, w, h).
            mixup_weight: the weight of mixup_feature, shape (b,). If one image in the batch does not apply to dhf, 
                        the entry for the image is 0.
            random_keep_prob: the probability one element in x is not perturbed.
            
        '''
        def uniform_random_like(x, minval, maxval):
            return (maxval - minval) * torch.rand_like(x, requires_grad=False) + minval

        dhf_indicator = self.dhf_indicator.reshape(-1, 1, 1, 1)  # shape (b, 1, 1, 1)
        # 1. mix up with mixup_feature
        mixup_weight = dhf_indicator * uniform_random_like(x, minval=0, maxval=self.mixup_weight_max)
        x = mixup_weight * self.mixup_feature + (1. - mixup_weight) * x
        # 2. randomly adjust some elements
        random_val = torch.mean(x, dim=(1, 2, 3), keepdim=True)  # shape (b, 1, 1, 1)
        x = torch.where((torch.rand_like(x)>=self.random_keep_prob)*(dhf_indicator>0), random_val, x)
        return x


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


def convert_to_DHF_model_inplace_(model, mixup_weight_max: float, random_keep_prob: float, dhf_modules):
    # This function convert a normal model to DHF model **inplace**
    for name in dhf_modules:
        old_layer = get_layer(model, name)
        new_layer = nn.Sequential(
            old_layer,
            DHFUnit(
                mixup_weight_max=mixup_weight_max, 
                random_keep_prob=random_keep_prob
                )
            )
        set_layer(model, name, new_layer)
    return model


def turn_on_dhf_update_mf_setting(model: nn.Module):
    for module in model.modules():
        if isinstance(module, DHFUnit):
            module.if_dhf = False
            module.update_mf = True
    model.eval()


def trun_off_dhf_update_mf_setting(model: nn.Module):
    for module in model.modules():
        if isinstance(module, DHFUnit):
            # module.if_dhf = True  # not necessary
            module.update_mf = False

def turn_on_dhf_attack_setting(model: nn.Module, dhf_indicator: Tensor):
    for module in model.modules():
        if isinstance(module, DHFUnit):
            module.if_dhf = True
            module.update_mf = False
            module.dhf_indicator = dhf_indicator

def preview_model(model: nn.Module):
    for name, module in model.named_modules():
        print("name: ", name, ", module", module)


# def visualize_model(model: nn.Module, input_size, filepath=""):
#     # if input_example is None:
#     #     input_example = torch.empty((10, 3, 224, 224))
#     # y = model(input_example)
#     # return make_dot(y[0].mean(), params=dict(model.named_parameters()))
#     model_graph = draw_graph(model, input_size=input_size, expand_nested=True)
#     return model_graph