import os
import glob
import xarray as xr
import numpy as np
import argparse

"""

Data Cleaning Pipeline for NAM and uWRF data

NAM: Can be downloaded from the NOAA website-- 
 https://www.ncei.noaa.gov/has/HAS.FileAppRouter?datasetname=NAM218&subqueryby=STATION&applname=&outdest=FILE

uWRF: Uploaded from UAlbany collaborators

The current state of this pipeline assumes the raw data has been uploaded to the correct
directory on BNL's remote server

difference between this and original is that we are not redoing the dimensions
this is for the Tristate area

""" 


class DataLoader:
    def __init__(self, dir_path, data_name, ds_var, pred_var=None):
        self.dir_path = dir_path
        self.data_name = data_name
        self.ds_var = ds_var #The variable we would like to downscale
        self.pred_var = pred_var #Optional: predictor variable
        self.dataset = None #This will store the loaded data
    
    def load_data(self):
        
        if self.pred_var is not None:
            print(f"Using {self.pred_var} as a predictor variable.")
        else:
            print("No predictor variable provided.")
            
        # Load the dataset (assumed to be in NetCDF format)
        self.dataset = xr.open_dataset(self.dir_path)
        return self.dataset

class uWRFDataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def rename_and_filter_vars_with_pred(self, dataset, ds_var, pred_var):
        
        #Check if the variables exist in the dataset
        #available_vars = set(data.variables.keys())
        #print(f"Available variables in the dataset: {available_vars}")
        
        #Only keep the ds_var and the pred_var
        dataset = dataset[[ds_var, pred_var]]
        dataset = dataset.rename({'XLAT': 'latitude', 'XLONG': 'longitude', 'XTIME': 'time'})
        return dataset
    
    def rename_and_filter_vars_no_pred(self, dataset, ds_var):
        dataset = dataset[[ds_var]]
        dataset = dataset.rename({'XLAT': 'latitude', 'XLONG': 'longitude', 'XTIME': 'time'})   
        return dataset
        

class NAMDataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def rename_and_filter_vars_with_pred(self, dataset, ds_var, pred_var):
        
        #Change how the longitude vales are measured
        dataset['longitude'] = (dataset['longitude'] + 180) % 360 - 180
        
        #Check if both of the variables exist in each dataset
        missing_vars = [var for var in [ds_var, pred_var] if var not in dataset]
        if missing_vars:
            print(f"Skipping dataset: Missing variables {missing_vars}")
            return None
        return dataset[[ds_var, pred_var]]

    def rename_and_filter_vars_no_pred(self, dataset, ds_var):
        dataset = dataset[[ds_var]]
        
        #Change how the longitude values are measured
        dataset['longitude'] = (dataset['longitude'] + 180) % 360 - 180
        return dataset

class NAMSpatialCutter:
    def __init__(self, dataset):
        self.dataset = dataset 
    
    def cut(self, dataset):
        # Bounds for y (latitude) and x (longitude) indices
        min_y, max_y = 73146, 524213  # y bounds measured in m
        min_x, max_x = 292584, 719269  # x bounds measured in m 
        
        # Extract the y and x indices
        y = dataset['y'].values  # 'y' corresponds to latitude
        x = dataset['x'].values  # 'x' corresponds to longitude
        
        # Get indices within the bounds
        y_indices = np.where((y >= min_y) & (y <= max_y))[0]
        x_indices = np.where((x >= min_x) & (x <= max_x))[0]
        
        # Slice the dataset using the indices
        dataset_cut = dataset.isel(y=y_indices, x=x_indices)
        
        return dataset_cut
    
class NAMInterpolator:
    def __init__(self, dataset, ds_var, pred_var):
        self.dataset = dataset
        self.ds_var = ds_var
        self.pred_var = pred_var
    
    #at this point, the dataset will be in order, for a whole month
    # the dimensions will be 
    def interpolator(self, dataset, ds_var, pred_var):
        uwrf_data = xr.open_dataset('/D4/data/gvaillant/uwrf-cleaned/uwrf_cleaned_01.nc')
        uwrf_shape = uwrf_data.dims #Time, south_north, west_east
        nam_shape = dataset.dims #y, x, time
        
        lat_ratio = uwrf_shape['south_north'] // nam_shape['y']
        lon_ratio = uwrf_shape['west_east'] // nam_shape['x']
        
        # Ensure time is aligned
        min_time_steps = min(uwrf_shape['time'], nam_shape['time'])
        uwrf_data = uwrf_data.isel(time=slice(0, min_time_steps))
        nam_data = dataset.isel(time=slice(0, min_time_steps))
        
        # Function to aggregate uWRF cells to NAM resolution
        def aggregate_uwrf(var):
            return var.coarsen(south_north=lat_ratio, west_east=lon_ratio, boundary="trim").mean()
    
        # Variables to align
        vars_to_align = [ds_var, pred_var]
        aligned_data = {}
    
        for var in vars_to_align:
            if var in uwrf_data:
                aligned_data[var] = aggregate_uwrf(uwrf_data[var])
    
        # Create new dataset with aligned variables
        aligned_nam = xr.Dataset(
            data_vars=aligned_data,
            coords={
                'y': nam_data.y,
                'x': nam_data.x,
                'time': nam_data.time
            },
            attrs=nam_data.attrs
        )
    
        return aligned_nam
    
def run_pipeline(dir_path, data_name, ds_var, pred_var=None):
    
    cleaned_uWRF_data = []
    cleaned_NAM_data = []
    
    if data_name == "uWRF":
        input_files = glob.glob(os.path.join(dir_path, '*'))
        for file in input_files:
            print(f"Processing file: {file}")
            loader = DataLoader(file, data_name, ds_var, pred_var)
            dataset = loader.load_data()
            cleaner = uWRFDataCleaner(dataset)
            
            if pred_var is not None:
                dataset = cleaner.rename_and_filter_vars_with_pred(dataset, ds_var, pred_var)
            else:
                dataset = cleaner.rename_and_filter_vars_no_pred(dataset, ds_var)
            
            cleaned_uWRF_data.append(dataset)
            
        final_dataset = xr.concat(cleaned_uWRF_data, dim = 'Time')
        
        #save to this directory
        #want to make the file name according to the month im doing
        output_dir = '/D4/data/gvaillant/uwrf-cleaned'
        month = os.path.basename(os.path.normpath(dir_path))
        
        output_file = os.path.join(output_dir, f"uwrf_cleaned_{month}.nc")
        
        final_dataset.to_netcdf(output_file)
        print(f"Saved cleaned uRWF dataset to: {output_file}")

        
    elif data_name == "NAM":
        input_files = glob.glob(os.path.join(dir_path, '*'))
        for file in input_files:
            print(f"Processing file: {file}")
            
            loader = DataLoader(file, data_name, ds_var, pred_var)
            dataset = loader.load_data()
            
            if dataset is None:
                print(f"Skipping file {file}: Dataset failed to load.")
                continue  # Skip to the next file
        
            cleaner = NAMDataCleaner(dataset)
            cutter = NAMSpatialCutter(dataset)
            #interpolation = NAMInterpolator(dataset, ds_var, pred_var)
            
            
            if pred_var is not None:
                dataset = cleaner.rename_and_filter_vars_with_pred(dataset, ds_var, pred_var)
            else:
                dataset = cleaner.rename_and_filter_vars_no_pred(dataset, ds_var)

                
            if dataset is not None:
                cleaned_NAM_data.append(dataset)
        
        if cleaned_NAM_data:
            
            final_dataset = xr.concat(cleaned_NAM_data, dim = 'time')
            final_dataset = cutter.cut(final_dataset)
        
            #final_dataset = interpolation.interpolator(final_dataset, ds_var, pred_var)
        
            #save to this directory
            #want to make the file name according to the month im doing
            output_dir = '/D4/data/gvaillant/NAM-cleaned'
            month = os.path.basename(os.path.normpath(dir_path))
        
            output_file = os.path.join(output_dir, f"nam_cleaned_{month}_test.nc")
            final_dataset.to_netcdf(output_file)
            
            print(f"Saved cleaned dataset to {output_file}")
        else:
            print("No valid datasets found. Skipping concatenation.")
        
    else:
        raise ValueError("Invalid data name specified. Either uWRF or  NAM.")



def main():
    parser = argparse.ArgumentParser(description="Run the data cleaning pipeline for uWRF or NAM data.")
    parser.add_argument("dir_path", type=str, help="Path to the directory containing input NetCDF files.")
    parser.add_argument("data_name", type=str, choices=["uWRF", "NAM"], help="Dataset name (uWRF or NAM).")
    parser.add_argument("ds_var", type=str, help="Variable to downscale.")
    parser.add_argument("--pred_var", type=str, default=None, help="Optional predictor variable.")

    args = parser.parse_args()

    run_pipeline(args.dir_path, args.data_name, args.ds_var, args.pred_var)

if __name__ == "__main__":
    main()



