import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def apply_initialization(m,
                         linear_mode="0",
                         conv_mode="0",
                         norm_mode="0",
                         embed_mode="0"):
    if isinstance(m, nn.Linear):

        if linear_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_in', nonlinearity="linear")
        elif linear_mode in ("1", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if conv_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if norm_mode in ("0", ):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    elif isinstance(m, nn.GroupNorm):
        if norm_mode in ("0", ):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    # # pos_embed already initialized when created
    elif isinstance(m, nn.Embedding):
        if embed_mode in ("0", ):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError
    else:
        pass