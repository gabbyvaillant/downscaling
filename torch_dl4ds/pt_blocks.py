import torch as pt
from .pt_utils import checkarg_dropout_variant
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(
            self,
            in_channels,    
            filters,
            strides=1,
            ks_cl1=(3, 3),
            ks_cl2=(3, 3),
            activation='relu',
            normalization=None,
            attention=False,
            dropout_rate=0,
            dropout_variant=None,
            depthwise_separable=False
    ):
        super().__init__()

        self.normalization = normalization
        self.attention = attention
        self.dropout_rate = dropout_rate
        self.dropout_variant = dropout_variant
        self.depthwise_separable = depthwise_separable
        self.in_channels = in_channels

        bias = normalization is None

        # Conv1
        if depthwise_separable:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=ks_cl1, padding='same',
                          stride=strides, groups=in_channels, bias=bias),
                nn.Conv2d(in_channels, filters, kernel_size=ks_cl1, stride=strides,
                          padding='same', bias=bias)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=ks_cl1,
                                   stride=strides, padding='same', bias=bias)

        # Conv2
        if depthwise_separable:
            self.conv2 = nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=ks_cl2, padding='same',
                          groups=filters, bias=bias),
                nn.Conv2d(filters, filters, kernel_size=1, bias=bias)
            )
        else:
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=ks_cl2,
                                   padding='same', bias=bias)

        # Normalization
        if normalization == 'bn':
            self.norm1 = nn.BatchNorm2d(filters)
            self.norm2 = nn.BatchNorm2d(filters)
        elif normalization == 'ln':
            self.norm1 = nn.LayerNorm([filters, 1, 1])  # Adjust shape during forward
            self.norm2 = nn.LayerNorm([filters, 1, 1])
        elif normalization is not None:
            raise ValueError(f"Unsupported normalization: {normalization}")

        # Attention
        if attention:
            self.att = ChannelAttention2D(filters)

        # Activation 
        if activation is None:
            self.activation_fn = None
        elif hasattr(F, activation):
            self.activation_fn = getattr(F, activation)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


        # Dropout
        if dropout_rate > 0:
            self.dropout1 = get_dropout_layer(dropout_rate, dropout_variant, dim=2)
            self.dropout2 = get_dropout_layer(dropout_rate, dropout_variant, dim=2)
            self.apply_dropout = True
        else:
            self.apply_dropout = False

    def forward(self, x):
        if self.apply_dropout:
            x = self.dropout1(x)
        x = self.conv1(x)
        if self.normalization:
            x = self.norm1(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)


        if self.apply_dropout:
            x = self.dropout2(x)
        x = self.conv2(x)
        if self.normalization:
            x = self.norm2(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)


        if self.attention:
            x = self.att(x)
        return x


class DropPath(nn.Module):
    """
    Drop path layer
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim - 1) # broadcastable shape
        random_tensor = keep_prob + pt.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = pt.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output
 
## --- Residual Block --- ##

class ResidualBlock(ConvBlock):
    """
    Residual block.

    Two exampleS:
    * Standard residual block [1]: Conv2D -> BN -> ReLU -> Conv2D -> BN -> Add -> ReLU
    * EDSR-style block: Conv2D -> ReLU -> Conv2D -> Add -> ReLU

    References
    ----------
    [1] Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
    """
    def __init__(
        self,
        in_channels,
        filters,
        strides=1,
        ks_cl1=(3,3),
        ks_cl2=(3,3),
        activation='relu',
        normalization=None,
        attention=False,
        dropout_rate=0,
        dropout_variant=None,
        use_1x1conv=False,
        depthwise_separable=False,
        **conv_kwargs):
        super().__init__(
            in_channels=in_channels,
            filters=filters,
            strides=strides,
            ks_cl1=ks_cl1,
            ks_cl2=ks_cl2,
            activation=activation,
            normalization=normalization,
            attention=attention,
            dropout_rate=dropout_rate,
            dropout_variant=dropout_variant,
            depthwise_separable=depthwise_separable,
            **conv_kwargs)
        
        self.use_1x1conv = use_1x1conv
        if self.use_1x1conv:
            self.conv1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=filters,
                kernel_size=1,
                stride=strides
            )

    def forward(self, x):
        identity = x

        out = x
        #Dropout before first conv if enabled
        if self.apply_dropout:
            out = self.dropout1(x)
        
        # First conv
        out = self.conv1(out)
        if self.normalization is not None:
            out = self.norm1(out)
        out = self.activation_fn(out)

        # (Optional) second dropout
        if self.apply_dropout:
            out = self.dropout2(out)

        # Second conv
        out = self.conv2(out)
        if self.normalization is not None:
            out = self.norm2(out)
        
        # (Optional) attention
        if self.attention:
            out = self.att(out)
        
        # (Optional) 1x1 conv to match shape
        if self.use_1x1conv:
            identity = self.conv1x1(identity)
        
        out = out + identity
        out = self.activation_fn(out)

        return out

class DenseBlock(ConvBlock):
    """
    Dense block.

    References
    ----------
    [1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Networks: 
        https://arxiv.org/abs/1608.06993
    """
    def __init__(
        self,
        in_channels,
        filters,
        strides=1,
        ks_cl1=(1,1),
        ks_cl2=(3,3),
        activation='relu',
        normalization=None,
        attention=False,
        dropout_rate=0,
        dropout_variant=None,
        depthwise_separable=False,
        **conv_kwargs):
        super().__init__(
            filters=filters,
            in_channels=in_channels,
            strides=strides,
            ks_cl1=ks_cl1,
            ks_cl2=ks_cl2,
            activation=activation,
            normalization=normalization,
            attention=attention,
            dropout_rate=dropout_rate,
            dropout_variant=dropout_variant,
            depthwise_separable=depthwise_separable,
            **conv_kwargs)
    
        # Override Conv1 to use bottleneck (1x1) with 4*filters
        bias = normalization is None
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels= 4 * filters,
            kernel_size=ks_cl1,
            stride=strides,
            padding='same',
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels=4 * filters,
            out_channels = filters,
            kernel_size = ks_cl2,
            stride = 1,
            padding = 'same',
            bias = bias
        )

        # Adjust norm layers if using normalization
        if normalization == 'bn':
            self.norm1 = nn.BatchNorm2d(4 * filters)
            self.norm2 = nn.BatchNorm2d(filters)
        elif normalization == 'ln':
            self.norm1 = nn.LayerNorm([4 * filters, 1, 1])
            self.norm2 = nn.LayerNorm([filters, 1, 1])
        
        if attention:
            self.att = ChannelAttention2D(filters)

    def forward(self, x):
        
        out = x

        # First norm -> act -> dropout -> conv1
        if self.normalization:
            out = self.norm1(out)
        out = self.activation_fn(out)
        if self.apply_dropout:
            out = self.dropout1(out)
        out = self.conv1(out)

        # Second norm -> act -> dropout -> conv2
        if self.normalization:
            out = self.norm2(out)
        out = self.activation_fn(out)
        if self.apply_dropout:
            out = self.dropout2(out)
        out = self.conv2(out)

        if self.attention:
            out = self.att(out)

        # Concatenate input and output along channels
        return pt.cat([x, out], dim=1)
    
class TransitionBlock(nn.Module):
    """
    TransitionBlock:

    Conv2D (1x1) -> Normalization (optional) -> Activation (optional)
    """
    def __init__(self, in_channels, out_channels=None, activation='relu', normalization=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=(normalization is None))
        self.normalization = normalization

        # Always define norm
        if normalization == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'ln':
            self.norm = nn.LayerNorm([out_channels, 1, 1])
        elif normalization is None:
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        if activation is None:
            self.activation_fn = None
        elif hasattr(F, activation):
            self.activation_fn = getattr(F, activation)
        else:
            raise ValueError(f"Unsupported activation: {activation}")


    def forward(self, x):
        x = self.conv(x)

        if self.norm is not None:
            if self.normalization == 'ln':
                b, c, h, w = x.shape
                x = self.norm(x.view(b, c, -1)).view(b, c, h, w)
            else:
                x = self.norm(x)
        
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x



class LocallyConnected2D_1x1(nn.Module):
    """
    1x1 Locally Connected Layer (no weight sharing)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            height,
            width,
            use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.use_bias = use_bias

        # One weight per spatial location, input channel, and output channel
        self.weight = nn.Parameter(
            pt.randn(height, width, in_channels, out_channels)
        )

        if use_bias:
            self.bias = nn.Parameter(
                pt.zeros(height, width, out_channels)
            )
        else:
            self.bias = None
    
    def forward(self, x):
        # x shape: [B, C_in, H, W]
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels
        assert H == self.height and W == self.width, " Input spatial dims must match layer initialization"

        # reshape x: [B, C_in, H, W] -> [B, H, W, C_in]
        x = x.permute(0, 2, 3, 1)

        # multiply with local weights: [B, H, W, C_in] @ [H, W, C_in, C_out]
        out = pt.einsum('bhwc,hwco->bhwo', x, self.weight)

        if self.use_bias:
            out += self.bias

        # reshape back: [B, H, W, C_out] -> [B, C_out, H, W]
        return out.permute(0, 3, 1, 2)
    
class LocalizedConvBlock(nn.Module):
    """ 
    Localized convolutional block through a locally connected layer (1x1 kernel) 
    with biases.
    """
    
    def __init__(
        self,
        in_channels,
        height,
        width,
        filters=2,
        activation=None,
        normalization=None,
        use_bias=True,
        name_suffix=''
    ):
        super().__init__()
        self.transition = TransitionBlock(in_channels=filters,  activation=activation, normalization=normalization)
        self.localconv = LocallyConnected2D_1x1(
            in_channels=in_channels,
            out_channels=filters,
            height=height,
            width=width,
            use_bias=use_bias
        )
        self.activation_fn = getattr(F, activation) if activation else None
    
    def forward(self, x):

        x = self.transition(x)

        x = self.localconv(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class SubpixelConvolutionBlock(nn.Module):
    """
    Subpixel convolution (pixel shuffle) block for upsampling.
    
    scale: overall upsampling factor (e.g., 2, 4, 10, 20)
    n_filters: number of output filters (after upsampling)

    References
    ----------
    [1] Real-Time Single Image and Video Super-Resolution Using an Efficient 
    Sub-Pixel Convolutional Neural Network: https://arxiv.org/abs/1609.05158
    """

    def __init__(
            self,
            scale,
            n_filters,
            in_channels,
            name_suffix=''):
        super().__init__()
        self.scale = scale
        self.n_filters = n_filters

        # 3x3 conv layers for different scale factors
        self.conv = nn.Conv2d(in_channels, n_filters * (scale ** 2), kernel_size=3, padding=1)
        self.conv2x = nn.Conv2d(in_channels, n_filters * (2 ** 2), kernel_size=3, padding=1)
        self.conv5x = nn.Conv2d(in_channels, n_filters * (5 **2), kernel_size=3, padding=1)

    def upsample_conv(self, x, factor):
        if factor == 2:
            x = self.conv2x(x)
            x = F.pixel_shuffle(x, upscale_factor=2)
            return x 
        elif factor == 5:
            x = self.conv5x(x)
            x = F.pixel_shuffle(x, upscale_factor=5)
            return x 
        else:
            x = self.conv(x)
            x = F.pixel_shuffle(x, upscale_factor=factor)
            return x


    def forward(self, x):
        if self.scale == 2:
            x = self.upsample_conv(x, 2)
        elif self.scale == 4:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
        elif self.scale == 8:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
        elif self.scale == 10:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 5)
        elif self.scale == 20:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 5)
        else:
            x = self.upsample_conv(x, self.scale)
        return x
    
class ChannelAttention2D(nn.Module):
    """
    Channel Attention mechanism from CBAM for 2D conv feature maps.

    Parameters:
        nf (int): Number of input channels (feature maps).
        r (int): Reduction ratio. Default is 4.

    Input:
        x (Tensor): [B, C, H, W]

    Output:
        Tensor: same shape [B, C, H, W] with channel attention applied
    """
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // r, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels // r, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        y = pt.mean(x, dim=(2 ,3), keepdim=True)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = pt.sigmoid(y)
        return x * y
    
class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, training=True)

class MCSpatialDropout2D(nn.Dropout2d):
    def forward(self, input):
        return F.dropout2d(input, self.p, training=True)

class MCSpatialDropout3D(nn.Dropout3d):
    def forward(self, input):
        return F.dropout3d(input, self.p, training=True)

def get_dropout_layer(dropout_rate, dropout_variant=None, dim=2):
    """
    Returns the appropriate dropout layer given the variant and dimensionality
    """
    if dropout_rate == 0:
        return nn.Identity()

    if dropout_variant is None or dropout_variant == 'vanilla':
        return nn.Dropout(p=dropout_rate)
    elif dropout_variant == 'gaussian':
        raise NotImplementedError("GaussianDropout is not natively supported in PyTorch")
    elif dropout_variant == 'spatial':
        if dim == 2:
            return nn.Dropout2d(p=dropout_rate)
        elif dim == 3:
            return nn.Dropout3d(p=dropout_rate)
    elif dropout_variant == 'mcdrop':
        return MCDropout(p=dropout_rate)
    elif dropout_variant == 'mcspatialdrop':
        if dim == 2:
            return MCSpatialDropout2D(p=dropout_rate)
        elif dim == 3:
            return MCSpatialDropout3D(p=dropout_rate)

    raise ValueError(f"Unsupported dropout variant: {dropout_variant}")