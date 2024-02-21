from functools import partial

import torch
import torch.nn as nn
import math
from einops import reduce, rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg

import torch.nn.functional as F

__all__ = [
    "tiny_patch16_224_hierarchical", "small_patch16_224_hierarchical", "base_patch16_224_hierarchical"
]


class TransformerHead(nn.Module):
    expansion = 1

    def __init__(self, token_dim, num_patches=196, num_classes=1000, stride=1):
        super(TransformerHead, self).__init__()

        self.token_dim = token_dim
        self.num_patches = num_patches
        self.num_classes = num_classes

        # To process patches
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.token_dim != self.expansion * self.token_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.token_dim, self.expansion * self.token_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.token_dim)
            )

        self.token_fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x):
        """
            x : (B, num_patches + 1, D) -> (B, C=num_classes)
        """
        cls_token, patch_tokens = x[:, 0], x[:, 1:]
        size = int(math.sqrt(x.shape[1]))

        patch_tokens = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=size, w=size)  # B, D, H, W
        features = F.relu(self.bn(self.conv(patch_tokens)))
        features = self.bn(self.conv(features))
        features += self.shortcut(patch_tokens)
        features = F.relu(features)
        patch_tokens = F.avg_pool2d(features, 14).view(-1, self.token_dim)
        cls_token = self.token_fc(cls_token)

        out = patch_tokens + cls_token

        return out


class VisionTransformer_hierarchical(VisionTransformer):
    def __init__(self, *args, **kwargs):
        if 'pretrained_cfg' in kwargs:
            kwargs.pop('pretrained_cfg')
        if 'pretrained_cfg_overlay' in kwargs:
            kwargs.pop('pretrained_cfg_overlay')
        super().__init__(*args, **kwargs)

        # Transformer heads
        self.transformerheads = nn.Sequential(*[
            TransformerHead(self.embed_dim)
            for i in range(11)])

    def forward_features(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Store transformer outputs
        transformerheads_outputs = []

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx <= 10:
                out = self.norm(x)
                out = self.transformerheads[idx](out)
                transformerheads_outputs.append(out)

        x = self.norm(x)
        return x, transformerheads_outputs

    def forward(self, x):
        x, transformerheads_outputs = self.forward_features(x)
        output = []
        for y in transformerheads_outputs:
            output.append(self.head(y))
        output.append(self.head(x[:, 0]))
        return output


@register_model
def tiny_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_tiny_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def small_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_small_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def base_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Improving-Adversarial-Transferability-of-Vision-Transformers"
                "/releases/download/v0/deit_base_trm.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    return model
