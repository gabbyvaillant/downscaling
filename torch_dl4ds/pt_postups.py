import torch as pt
import torch.nn as nn
from .pt_blocks import (
    ResidualBlock, ConvBlock, DenseBlock, TransitionBlock, SubpixelConvolutionBlock,
    LocalizedConvBlock, get_dropout_layer
)
from .pt_utils import (checkarg_backbone, checkarg_upsampling, 
                    checkarg_dropout_variant)


class PostUpsamplingNet(nn.Module):
    def __init__(
            self,
            backbone_block,
            upsampling,
            scale,
            n_channels,
            lr_size,
            n_channels_out=1,
            n_filters=8,
            n_blocks=6,
            normalization=None,
            dropout_rate=0,
            dropout_variant=None,
            attention=False,
            activation='relu',
            output_activation=None,
            rc_interpolation='bilinear',
            localcon_layer=False
    ):
        super().__init__()

        self.backbone_block = checkarg_backbone(backbone_block)
        self.upsampling = checkarg_upsampling(upsampling)
        self.dropout_variant = checkarg_dropout_variant(dropout_variant)

        self.scale= scale
        self.localcon_layer = localcon_layer
        self.activation = activation
        self.output_activation = output_activation

        h_lr, w_lr = lr_size
        h_hr = int(h_lr * scale)
        w_hr = int(w_lr * scale)

        init_n_filters = n_filters
        self.ks = 3

        #Initial Conv
        self.initial_conv = nn.Conv2d(n_channels, n_filters, kernel_size=self.ks, padding=self.ks // 2)

        # Backbone blocks
        blocks = []
        for i in range(n_blocks):
            in_channels = n_filters if i == 0 else init_n_filters * i
            out_channels = init_n_filters * (i + 1)

            if backbone_block == 'resnet':
                blocks.append(ResidualBlock(
                    filters=out_channels,
                    in_channels=in_channels,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    dropout_variant=dropout_variant,
                    normalization=normalization,
                    use_1x1conv=False if i == 0 else True,
                    attention=attention
                ))
            elif backbone_block == 'densenet':
                blocks.append(DenseBlock(
                    filters=out_channels,
                    in_channels=in_channels, 
                    activation=activation,
                    dropout_rate=dropout_rate,
                    dropout_variant=dropout_variant,
                    normalization=normalization,
                    attention=attention
                ))
                blocks.append(TransitionBlock(out_channels // 2))
                out_channels = out_channels // 2  # update for next block

        #print(blocks)
        self.backbone = nn.Sequential(*blocks)
        #print(self.backbone)

        #Upsampling
        if upsampling == 'spc':
            self.upsample = SubpixelConvolutionBlock(
                scale,
                n_filters=out_channels,
                in_channels=out_channels
                )
            
        #Local conv
        if localcon_layer:
            self.localconv = LocalizedConvBlock(
                in_channels=out_channels,
                height=h_hr,
                width=w_hr,
                filters=out_channels,
                activation=activation,
                normalization=normalization,
                use_bias=True
                )

        # Final Layers
        # Start with upsampled output
        transition_in_channels = out_channels

        if localcon_layer:
            transition_in_channels += out_channels

        self.transition_last = TransitionBlock(
            filters=transition_in_channels,
            activation=activation,
            normalization=normalization
        )


        self.final_block1 = ConvBlock(
            in_channels=transition_in_channels,
            filters=init_n_filters,
            ks_cl1=self.ks,
            ks_cl2=self.ks,
            activation=None,
            dropout_rate=dropout_rate,
            normalization=normalization,
            attention=True
        )

        self.final_block2 = ConvBlock(
            in_channels=init_n_filters,
            filters=n_channels_out,
            ks_cl1=self.ks,
            ks_cl2=self.ks,
            activation=output_activation,
            dropout_rate=0,
            normalization=normalization,
            attention=False
        )


    def forward(self, x_in, s_in=None):
        x = self.initial_conv(x_in)
        backbone_output = self.backbone(x)
        x = backbone_output
        x = self.upsample(x)

        if self.localcon_layer:
            lws = self.localconv(x)
            x = pt.cat([x, lws], dim=1)

        x = self.transition_last(x)
        x = self.final_block1(x)
        x = self.final_block2(x)

        return x

def net_postupsampling(*args, **kwargs):
    return PostUpsamplingNet(*args, **kwargs)