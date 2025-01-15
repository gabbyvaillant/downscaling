# Deep Learning-Based Downscaling of Low-Resolution Weather Forecast Data for New York City 🌧️

## Overview
This repository contains the code for applying dl4ds, an open-source deep learning library for climate downscaling, to forecast model outputs. The project focuses on leveraging deep learning to enhance the resolution of weather forecast data, making it actionable and precise for localized areas.

## Data 📊
High-Resolution Data: Ground truth data for model training is sourced from the urbanized Weather Research and Forecasting (uWRF) model, developed by collaborators at the University at Albany. This dataset provides 3 km resolution and 3-hourly temporal granularity.

Low-Resolution Data: The North American Mesoscale (NAM) model, with a resolution of 12 km and 3-hourly intervals, serves as the input dataset for downscaling.

## Goal 🎯

We apply our deep learning model to transform the coarse-resolution NAM data into high-resolution weather forecasts, specifically tailored for New York City. This effort enables better urban-scale forecasting to support decision-making for energy system planners.

## Directory Overview 📂
**data**

Contains data needed to run models

**models**

Contains three different deep learning models.

**prep**

Contains all preprocessing code needed before inputting data into models.




Currently downscaling temperature at 2m with surface pressure as a predictor variable (model takes this as another input). The goal is to downscale a variable such as wind direction, solar irrandiance, wind speed.

