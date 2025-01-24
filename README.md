# Deep Learning-Based Downscaling of Low-Resolution Weather Forecast Data for New York City 🌧️

## Overview
This repository hosts the code for utilizing [**dl4ds**](https://github.com/carlos-gg/dl4ds), an open-source deep learning library designed for climate downscaling, to process forecast model outputs. The project aims to employ deep learning techniques to improve the resolution of weather forecast data, enhancing its accuracy and usability for precise locations.

## Data 📊
High-Resolution Data: Ground truth data for model training is sourced from the urbanized Weather Research and Forecasting (uWRF) model, developed by collaborators at the University at Albany. This dataset provides 3 km resolution and 3-hourly temporal granularity.

Low-Resolution Data: The North American Mesoscale (NAM) model, with a resolution of 12 km and 3-hourly intervals, serves as the input dataset for downscaling.

## Goal 🎯

We apply our deep learning model to transform the coarse-resolution NAM data into high-resolution weather forecasts, specifically tailored for New York City. This effort provides access to more high resolution data for energy system planning.

## Directory Overview 📂
**cleaning**

Notebooks used to clean data.

**data**

TODO: add other data files and better labeling

**models**

Contains four deep learning models.
TODO: Tune hyperparams

**results**

Information about test loss and learning curves.



