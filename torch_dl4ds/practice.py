import torch.nn as nn
import torch
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(input.shape)
print(output.shape)

## extra

import torch as pt
import torch.nn as nn

class ConvNextBlock(nn.Module):
    """
    ConvNext block.

    References
    ----------
    [1] A ConvNet for the 2020s: https://arxiv.org/abs/2201.03545 
    """
    def __init__(
            self,
            filters,
            drop_path=0.,
            layer_scale_init_value=0., #1e-6
            use_1x1conv=False,
            activation='gelu',
            normalization='ln',
            name=None,
            **conv_kwargs):
        
        super().__init__()

        self.filters = filters
        self.use_1x1conv = use_1x1conv
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Depthwise Conv2D with kernel_size=7 and padding='same'
        self.dwconv = nn.Conv2d(filters, filters, kernel_size=7, padding=3, groups=filters)

        # Normalization layer
        if normalization == 'ln':
            self.norm = nn.LayerNorm(filters, eps=1e-6)
        elif normalization == 'bn':
            self.norm = nn.BatchNorm2d(filters)
        else:
            raise ValueError(f'Unsupported normalization: {normalization}')
        
        #go over again
        #Pointwise MLP (pwconv1, activation, pwconv2)
        self.pwconv1 = nn.Linear(filters, 4 * filters)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.pwconv2 = nn.Linear(4 * filters, filters)

        # Optional 1x1 conv if input shape mismatch
        self.pwconv1 = nn.Conv2d(filters, filters, kernel_size=1) if use_1x1conv else nn.Identity()

        # Optional layer scale parameter
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * pt.ones((filters)))
        else:
            self.gamma = None

    def forward(self, x):
        input = x #residual

        x = self.dwconv(x) # [B, C, H, W]

        if isinstance(self.norm, nn.LayerNorm):
            # LayerNorm expect shape [B, C, H, W] -> [B, H, W, C]
            x = x.permute(0, 2, 3, 1) # -> [B, H, W, C]
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # -> [B, C, H, W]
        else:
            #BatchNorm2d path
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)
            x = self.pwconv1(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2 )

        # Optional 1x1 conv if enabled
        shortcut = self.conv1x1(input)

        # Residual + drop path
        return shortcut + self.drop_path(x)
    