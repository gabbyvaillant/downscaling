import torch
from pt_postups import PostUpsamplingNet

# Dummy input settings
batch_size = 2
n_channels = 3
n_aux_channels = 2
lr_height = 30
lr_width = 30
scale = 2
hr_height = lr_height * scale
hr_width = lr_width * scale

# Create dummy inputs
x_in = torch.randn(batch_size, n_channels, lr_height, lr_width)
s_in = torch.randn(batch_size, n_aux_channels, hr_height, hr_width)

# Initialize model
model = PostUpsamplingNet(
    backbone_block='resnet',
    upsampling='spc',
    scale=scale,
    n_channels=n_channels,
    n_aux_channels=n_aux_channels,
    lr_size=(lr_height, lr_width),
    n_channels_out=1,
    n_filters=7,
    n_blocks=3,
    normalization='bn',
    dropout_rate=0.1,
    dropout_variant='vanilla',
    attention=True,
    output_activation=None,
    localcon_layer=True
)

# Forward pass
out = model(x_in, s_in)
print(f"Output shape: {out.shape}")
