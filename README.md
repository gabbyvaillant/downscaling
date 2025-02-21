# Deep Learning-Based Downscaling of Low-Resolution Weather Forecast Data for New York City 🌧️

## Overview
Welcome to the downscaling repository! This repository uses [**dl4ds**](https://github.com/carlos-gg/dl4ds), an open-source deep learning library designed for climate downscaling, to process forecast model outputs. Our goal is to use deep learning techniques to improve the spatial and temporal resolution of weather forecast data, enhancing its accuracy and usability.

## Data 📊
High-Resolution Data: Ground truth data for model training is sourced from the urbanized Weather Research and Forecasting (uWRF) model, developed by collaborators at the University at Albany. This dataset provides 3 km resolution and 3-hourly temporal granularity.

Low-Resolution Data: The North American Mesoscale (NAM) model, with a resolution of 12 km and 3-hourly intervals, serves as the input dataset for downscaling.

## Goal 🎯

We apply a deep learning model to transform the coarse-resolution NAM forecasts into high-resolution weather forecasts, specifically tailored for New York City tristate area. At it's current stage, the model only downscales NAM data spatially (12km to 3km). We eventually hope to downscale NAM temporally (3-hourly to hourly). This effort provides access to more high resolution data that can be used for energy system planning.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/gabbyvaillant/downscaling.git
   
   cd downscaling
   ```

2. Create a new virtual enviornment using Python 3.8.18

```
conda create -n downscaling python=3.8.18

conda activate myenv

#Install necessary libraries for project
#There may be errors with the dl4ds library bc I had to change the library manually (sklearn was outdated)

pip install -r requirements.txt

```

IF there is an error with installing dl4ds, delete that line from requirements.txt
cd into the dl4ds directory and do ``` pip install .``` which should correctly install dl4ds into the virtual enviornment.
Then, try ``` pip install -r requirements.txt  ``` again. 


3. Run downscaling model for Temperature on the NYC Tristate area

The notebook for this step is found here: /downscaling/training/T2-Tristate-Model.ipynb
Open the jupyter notebook and follow the instructions in the notebook to run the model.


 
