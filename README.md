# **Climate Downscaling** 🌧️

## Introduction
This repository contains code developed for climate downscaling research using machine learning techniques. The objective is to utilize the TimeGAN model for downscaling selected climate datasets (GCMs in netCDF file format). The project initially focuses on generating synthetic weather data and aims to implement climate downscaling methodologies within the TimeGAN framework.

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
    - Temperature 2m above ground (NAM: TMP_2maboveground, uWRF: T2)
    - U-component of wind (NAM: UGRD_10maboveground, uWRF: U10)
    - V-component of wind (NAM: VGRD_10maboveground, UWRF: V10)
    - Surface pressure (NAM: PRES_surface, UWRF: PSFC)
