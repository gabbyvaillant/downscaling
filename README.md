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
   git clone [https://github.com/gabbyvaillant/downscaling](https://github.com/gabbyvaillant/downscaling.git)
   cd downscaling
   ```

2. Create a new virtual enviornment using Python 3.8.18

 Using Python 3.8.18
 
