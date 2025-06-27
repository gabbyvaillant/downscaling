# torch_dl4ds/config.py

BACKBONE_BLOCKS = [
    'resnet',
    'densenet',
]

UPSAMPLING_METHODS = [
    'spc',
]

POSTUPSAMPLING_METHODS = ['spc']

INTERPOLATION_METHODS = [
    'inter_area',
]

LOSS_FUNCTIONS = [
    'mae',
]

DROPOUT_VARIANTS = [
    'vanilla',
    'gaussian',
    'spatial',
    'spatialdrop',
    'mcgaussiandrop',
    'mcspatialdrop'
]
