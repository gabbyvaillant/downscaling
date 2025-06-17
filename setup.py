from setuptools import setup, find_packages

setup(
    name='torch-dl4ds',
    version='1.8.0',
    description='PyTorch version of DL4DS (Deep Learning for Downscaling)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Gabrielle Vaillant',
    author_email='gvaillant1@bnl.gov',
    packages=find_packages(),  # Automatically finds `torch_dl4ds` if it's in a folder
    install_requires=[
        'torch',
        'numpy',
        'xarray',
        'scikit-learn',
        'matplotlib',
        # Add more if needed
    ],
    python_requires='>=3.8',
)
