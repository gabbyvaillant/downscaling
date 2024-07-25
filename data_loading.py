"""Climate downscaling applied to TimeGAN


Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

TimeGAN Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks


- Editing the functions found in data_loading.py to work for climate downscaling

-----------------------------

data_loading.py

(1) MinMaxScaler: Custom Min Max normalizer
(2) netCDF_data_loading: Load and preprocess climate data (NetCDF file format)

Examples found at end of script

"""

## Import Necessary Packages

import xarray as xr
import numpy as np

###############################################################

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data

###############################################################

def netCDF_data_loading(nc_file, var_names, seq_len, dims):
    """Reads NetCDF file and converts to a pandas DataFrame.
    
    Args:
    nc_file (str): Path to the climate data (NetCDF file).
    var_names (list of str): List of variable names to downscale from nc_file.
    seq_len (int): Sequence list.
    dims (list of str): List of the dimensions from tnc_file.
    
    Returns:
    data (list of np.ndarray): List of preprocessed sequences.
    """
    
    # Read the NetCDF file using xarray
    ds = xr.open_dataset(nc_file)
    
    
    # Check if the variables exist in the dataset
    for var_name in var_names:
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in the dataset.")
    
    
    # Convert each variable to DataFrame and merge them
    dfs = []
    for var_name in var_names:
        df = ds[var_name].to_dataframe().reset_index()
        df = df.rename(columns={var_name: var_name})
        dfs.append(df)
    
    # Merge DataFrames on 'time', 'lat', and 'lon'
    merged_df = dfs[0]
    for df in dfs[1:]:
        
        #NOTE: Change the below line depending on the index of your .nc file
        #merged_df = merged_df.merge(df, on=['time', 'lat', 'lon'], how='inner')
        merged_df = merged_df.merge(df, on = dims, how = 'inner')
    
        # Normalize the data
    for column in var_names:
        merged_df[column] = MinMaxScaler(merged_df[column])

    # Convert DataFrame to numpy array for sequence splitting
    orig_data = merged_df[var_names].values
    
    # Preprocess the dataset
    temp_data = []    
  
    # Cut data by sequence length
    for i in range(0, len(orig_data) - seq_len):
        _x = orig_data[i:i + seq_len]
        temp_data.append(_x)
        
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))    
    data = [temp_data[i] for i in idx]
    
    return data

###############################################################

"""Example 1

Info: Daily avg surface temp (tas) and relative humidity (hurs) dataset for 2015
Dimensions: time, lat, lon
NOTE: 
    - This example is included because this dataset includes data for every day whereas
the dataset we aim to downscale is inconsistent.
    - Change the merged_df line in the netCDF_data_loading function to the correct index


"""

#Specifying args for function
nc_file = '/Users/gabbyvaillant/EDA-MRI-ESM/source_gcm_data/temp_humi_day_MRI-ESM2-0_ssp585_r1i1p1f1_gn_20150101-20151231.nc'
var_names = ['tas', 'hurs']
seq_len = 7
dims = ['time', 'lat', 'lon']

data = netCDF_data_loading(nc_file, var_names, seq_len, dims)

print("Length of first example:")
print(len(data))

#print("First Sequence:")
#print(data[0])



""""Example 2

Info: NAM-NMM data includes 445 variables and is representing a 3 hour time slice ? of 02/20/2019
Dimensions: y, x, time = 1

NOTE: 
    - Change the merged_df line in the netCDF_data_loading function to the correct index

"""

#Specifying args for function
nc_file2 = '/Users/gabbyvaillant/Downloads/BNL/NAM2019/domnys-nam_218_20191011_0000_000.nc'
var_names = ['TMP_1000mb', 'RH_1000mb', 'VVEL_1000mb']
seq_len = 7
dims = ['x', 'y', 'time']

data = netCDF_data_loading(nc_file2, var_names, seq_len, dims)

print("Length of second example:")
print(len(data))

###############################################################





