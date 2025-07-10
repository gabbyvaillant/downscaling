![Downscaling Results](visuals/new-t2.png)

# Machine Learning-Based Downscaling of Low-Resolution Weather Forecast Data for New York City 🌧️

## Overview

Welcome to the downscaling repository! This repository uses an updated PyTorch version of the [**dl4ds**](https://github.com/carlos-gg/dl4ds) library. dl4ds is an open-source deep learning library designed for  downscaling climate datasets. Our specific goal is to use machine learning improve the spatial and temporal resolution of weather forecast data to enhance its accuracy and usability. 


## Directory Overview


(1) cleaning
Contains code for cleaning both low-resolution (NAM output) and high-resolution (uWRF output) data.

(2) dl4ds
A forked version of the original DL4DS library.

(3) make-predictions
Code for generating predictions on unseen data.

(4) pretrained-models
Saved downscaling models.

(5) results 
CSV files storing model metadata, tested hyperparameters, and their associated performance metrics.

(6) torch_dl4ds
PyTorch implementation of the DL4DS library.

(7) training
Notebooks and scripts for training the downscaling models. Each notebook focuses on a specific variable.

(8)
Visualizations of downscaled results.

(9) requirements.txt
Lists the required Python libraries to set up the virtual environment for torch_dl4ds.

(10) setup.py
Setup script for packaging and installing the library.

## Goal 🎯

We apply a deep learning model to transform the coarse-resolution NAM forecasts into high-resolution weather forecasts, specifically tailored for New York City tristate area. At it's current stage, the model only downscales NAM data spatially (12km to 3km). We eventually hope to downscale NAM temporally (3-hourly to hourly). This effort provides access to more high resolution data that can be used for energy system planning. 

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/gabbyvaillant/downscaling.git
   
   cd downscaling
   ```

2. Create a new virtual enviornment using Python 3.11.13

```
conda create -n downscaling python= 3.11.13

conda activate downscaling

```

3. Install the necessary libraries

```bash

pip install -r requirements.txt

```

3. Run downscaling model for Temperature on the NYC Tristate area

```bash

cd /downscaling/training

#Open Jupyter
jupyter lab
```

4. Check results

In the results directory, there is a .csv holding information about the different models that were tested. Here we can compare the loss and the training time.

## Data 📊

High-Resolution Data: Ground truth data for model training is sourced from the urbanized Weather Research and Forecasting (uWRF) model, developed by collaborators at the University at Albany. This dataset provides 3 km resolution and 3-hourly temporal granularity.

Low-Resolution Data: The North American Mesoscale (NAM) model, with a resolution of 12 km and 3-hourly intervals, serves as the input dataset for downscaling