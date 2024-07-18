# **Climate Downscaling**

## Introduction
This repository contains code developed for climate downscaling research using machine learning techniques. The objective is to utilize the TimeGAN model for downscaling selected climate datasets (GCMs in netCDF file format). The project initially focuses on generating synthetic weather data and aims to implement climate downscaling methodologies within the TimeGAN framework. I use the ydata-synthetic library to do this because it is an updated version of the TimeGAN framework. 

## Code Explanation

(0) data-manip directory

(a) netCDF_to_timeseries.ipynb
* Converted temperature netCDF file to .csv file for only ONE grid location
* Followed [Youtube tutorial](https://www.youtube.com/watch?v=hrm5RmsVXo0)
  
(1) weather-to-GAN.ipynb
* Trying to input a simple weather data file (.csv format) into the TimeGAN model using ydata-synthetic library
* Following tutorial used in [ydata-synthetic repository](https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb)

(2) data_loading.py

* Transform climate data in netCDF format to preprocessed data (normalized and split into squences)
* Generate Sine data (FIX LATER)
