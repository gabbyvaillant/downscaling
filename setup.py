import os
from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

setup(
    name='torch_dl4ds',
    version='1.0.0',
    description='PyTorch version of DL4DS (Deep Learning for Downscaling)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Gabrielle Vaillant',
    author_email='gvaillant1@bnl.gov',
    packages=find_packages(include=['torch_dl4ds', 'torch_dl4ds.*']),
    install_requires=[
        'torch',
        'numpy',
        'xarray',
        'scikit-learn',
        'scipy',
        'ecubevis',
        'matplotlib',
    ],
    python_requires='>=3.8',
)

