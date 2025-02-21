# Deep Learning-Based Downscaling of Low-Resolution Weather Forecast Data for New York City 🌧️

## Overview
Welcome to the downscaling repository! This repository uses [**dl4ds**](https://github.com/carlos-gg/dl4ds), an open-source deep learning library designed for climate downscaling, to process forecast model outputs. Our goal is to use deep learning techniques to improve the spatial and temporal resolution of weather forecast data, enhancing its accuracy and usability.

## Data 📊
High-Resolution Data: Ground truth data for model training is sourced from the urbanized Weather Research and Forecasting (uWRF) model, developed by collaborators at the University at Albany. This dataset provides 3 km resolution and 3-hourly temporal granularity.

Low-Resolution Data: The North American Mesoscale (NAM) model, with a resolution of 12 km and 3-hourly intervals, serves as the input dataset for downscaling.

## Goal 🎯

We apply a deep learning model to transform the coarse-resolution NAM forecasts into high-resolution weather forecasts, specifically tailored for New York City tristate area. At it's current stage, the model only downscales NAM data spatially (12km to 3km). We eventually hope to downscale NAM temporally (3-hourly to hourly). This effort provides access to more high resolution data that can be used for energy system planning.

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/gabbyvaillant/downscaling.git
   
   cd downscaling
   ```

2. Create a new virtual enviornment using Python 3.8.18

```
conda create -n downscaling python=3.8.18

conda activate downscaling

```

3. Install the necessary libraries

```bash

pip install jupyter
pip install xarray
pip install numpy
pip install ecubevis 
pip install scipy
pip install netCDF4

```

Once those libraries are installed, we must install the deep learning library, dl4ds. We must manually install this library. 

The following repository has an updated version of the dl4ds library:

```bash
git clone https://github.com/subhrajitjubu/dl4ds.git

#Once the dl4ds repository is cloned, cd into it while still in the virtual enviornment
cd dl4ds

#This will install the correct version of dl4ds
pip install .
```

3. Run downscaling model for Temperature on the NYC Tristate area

```bash

cd /downscaling/training

#Open Jupyter

jupyter lab
```

Once jupyter opens, go to the T2-Tristate-Model.ipynb notebook.
There you can begin running the cells. The first cell imports the necessary libraries. It will throw an error if we are missing some. Install the missing libraries into the virtual environment using pip

In the following cells, you will have to edit the paths to the data files. There are comments indicating which paths you should change. Once the paths are updated, then the notebook should run.

4. Check results

In the results directory, there is a .csv holding information about the different models that were tested. Here we can compare the loss and the training time.



 
