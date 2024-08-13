# **Climate Downscaling** 🌧️

## Introduction
This repository contains code developed for climate downscaling research using machine learning techniques.

## Code Explanation

(0) data-manip directory

(a) netCDF_to_timeseries.ipynb
* Converted temperature netCDF file to .csv file for only ONE grid location
* Followed [Youtube tutorial](https://www.youtube.com/watch?v=hrm5RmsVXo0)

(1) preprocessing directory

(a) preprocess-NAM
* Steps taken to preprocess NAM dataset to be consistent with uWRF dataset

(b) preprocess-uWRF
* Steps taken to preprocess uWRF dataset to be consistent with NAM dataset

(2) timeGAN-funcs
* Functions updated and edited to use from timeGAN


  ## Info:
  * Variables of interest:
    - T2: Temperature 2m above ground
    - U10: U-component of wind
    - V10: V-component of wind
    - PSFC: Surface pressure
