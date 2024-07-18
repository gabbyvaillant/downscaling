"""Climate downscaling applied to TimeGAN

Applying climate downscaling aspect to the TimeGAN model developed by Jinsung Yoon

I am using the same format and editing the functions from their utils.py

TimeGAN paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks


-----------------------------

preprocessing.py

(0) MinMaxScaler: Custom Min Max normalizer
(1) sine_data_generation: Generate sine dataset (EDIT THIS FUNCTION LATER)
(2) netCDF_data_loading: Load and preprocess climate data (NetCDF file format)
(3) Example usage
"""

## Necessary Packages

import xarray as xr
import numpy as np


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



def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data


def netCDF_data_loading(nc_file, var_names, seq_len):
    """Reads NetCDF file and converts to a pandas DataFrame.
    
    Args:
    nc_file (str): Path to the climate data (NetCDF file).
    var_names (list of str): List of variable names to downscale from nc_file.
    seq_len (int): Sequence list.
    
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
        merged_df = merged_df.merge(df, on=['time', 'lat', 'lon'], how='inner')
        
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



"""Example 1

Info: Daily avg surface temp (tas) and relative humidity (hurs) dataset for 2015
Dimensions: time, lat, lon


"""

# Specify arguments for function
nc_file = '/Users/gabbyvaillant/EDA-MRI-ESM/source_gcm_data/temp_humi_day_MRI-ESM2-0_ssp585_r1i1p1f1_gn_20150101-20151231.nc'
var_names = ['tas', 'hurs']
seq_len = 10

data = netCDF_data_loading(nc_file, var_names, seq_len)

print("Length:")
print(len(data))

print("First Sequence:")
print(data[0])



""""Example 2

Info: NAM-NMM data includes 445 variables and is representing a 3 hour time slice ? of 02/20/2019
Dimensions: y, x, time = 1

NOTE: since the dimensions are different names we need to change the function

"""

#SELECT VARS:
#netCDF_data_loading('/Users/gabbyvaillant/Downloads/BNL/NAM2019/domnys-nam_218_20190220_0000_000.nc', 2)





