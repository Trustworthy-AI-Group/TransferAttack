import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models import Inception3, Inception_V3_Weights
import warnings
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple
from timm.models.registry import register_model
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface



from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import  Weights, WeightsEnum
from timm.models.registry import register_model
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

__all__ = ["Inception3", "InceptionOutputs", "_InceptionOutputs", "Inception_V3_Weights", "inception_v3"]


InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


class GhostInception3(Inception3):
    def __init__(self, ghost_keep_prob=0.994, num_classes: int = 1000, aux_logits: bool = True, transform_input: bool = False, inception_blocks: Optional[List[Callable[..., nn.Module]]] = None, init_weights: Optional[bool] = None, dropout: float = 0.5) -> None:
        super().__init__(num_classes, aux_logits, transform_input, inception_blocks, init_weights, dropout)
        self.ghost_keep_prob = ghost_keep_prob

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.inc_v3.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        p = 1. - self.ghost_keep_prob
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        x = F.dropout(x, p=p)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        x = F.dropout(x, p=p)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        x = F.dropout(x, p=p)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        x = F.dropout(x, p=p)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        x = F.dropout(x, p=p)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        x = F.dropout(x, p=p)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        x = F.dropout(x, p=p)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        x = F.dropout(x, p=p)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        x = F.dropout(x, p=p)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        x = F.dropout(x, p=p)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        x = F.dropout(x, p=p)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        x = F.dropout(x, p=p)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        x = F.dropout(x, p=p)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        x = F.dropout(x, p=p)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        x = F.dropout(x, p=p)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        x = F.dropout(x, p=p)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        x = F.dropout(x, p=p)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        x = F.dropout(x, p=p)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

@register_model
@handle_legacy_interface(weights=("pretrained", Inception_V3_Weights.IMAGENET1K_V1))
def ghost_inception_v3(*, ghost_keep_prob=0.994, weights: Optional[Inception_V3_Weights] = None, progress: bool = True, **kwargs: Any) -> GhostInception3:
    """
    Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        weights (:class:`~torchvision.models.Inception_V3_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.Inception_V3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.Inception3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Inception_V3_Weights
        :members:
    """
    weights = Inception_V3_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", True)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GhostInception3(ghost_keep_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model
