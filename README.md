# Deep Learning-Based Downscaling of Low-Resolution Weather Forecast Data for New York City 🌧️

## Overview
This repository contains the code for applying dl4ds, an open-source deep learning library for climate downscaling, to forecast model outputs. The project focuses on leveraging deep learning to enhance the resolution of weather forecast data, making it actionable and precise for localized areas.

## Data
High-Resolution Data: Ground truth data for model training is sourced from the urbanized Weather Research and Forecasting (uWRF) model, developed by collaborators at the University at Albany. This dataset provides 3 km resolution and 3-hourly temporal granularity.

Low-Resolution Data: The North American Mesoscale (NAM) model, with a resolution of 12 km and 3-hourly intervals, serves as the input dataset for downscaling.

## Goal

We apply our deep learning model to transform the coarse-resolution NAM data into high-resolution weather forecasts, specifically tailored for New York City. This effort enables better urban-scale forecasting to support decision-making in areas such as disaster management, transportation, and environmental monitoring.

## Directory Overview

(1) preprocessing
