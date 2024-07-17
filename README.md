# **Climate Downscaling**

## Introduction
Repository for code created for my climate downscaling research. The goal of this project is to perform machine learning-based climate downscaling on our desired climate datasets. Currently, I am focused on using the TimeGAN model to accept weather data files (netCDF) and generate synthetic data. After that, I will focus on implementing the climate downscaling aspect to this TimeGAN model. 

## Code Explanation

(1) weather-to-GAN.ipynb
* Following tutorial used in [ydata-synthetic repository](https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb)
* Trying to input a simple weather data file (.csv format) into the TimeGAN model using ydata-synthetic library
