import os
import glob
import xarray as xr
import numpy as np
import argparse

class DataLoader:
    def __init__(self, dir_path, data_name, ds_vars, pred_var=None):
        self.dir_path = dir_path
        self.data_name = data_name
        self.ds_vars = ds_vars
        self.pred_var = pred_var
        self.dataset = None
    
    def load_data(self):
        if self.pred_var is not None:
            print(f"Using {self.pred_var} as a predictor variable.")
        else:
            print("No predictor variable provided.")
        self.dataset = xr.open_dataset(self.dir_path)
        return self.dataset

class uWRFDataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def rename_and_filter_vars(self, dataset, ds_vars, pred_var):
        vars_to_keep = list(ds_vars)
        if pred_var:
            vars_to_keep.append(pred_var)
        dataset = dataset[vars_to_keep]
        dataset = dataset.rename({'XLAT': 'latitude', 'XLONG': 'longitude', 'XTIME': 'time'})
        return dataset

class NAMDataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def rename_and_filter_vars(self, dataset, ds_vars, pred_var):
        dataset['longitude'] = (dataset['longitude'] + 180) % 360 - 180
        vars_to_keep = list(ds_vars)
        if pred_var:
            vars_to_keep.append(pred_var)
        missing_vars = [var for var in vars_to_keep if var not in dataset]
        if missing_vars:
            print(f"Skipping dataset: Missing variables {missing_vars}")
            return None
        return dataset[vars_to_keep]

class NAMSpatialCutter:
    def __init__(self, dataset):
        self.dataset = dataset 
    
    def cut(self, dataset):
        min_y, max_y = 73146, 524213
        min_x, max_x = 292584, 719269
        y = dataset['y'].values
        x = dataset['x'].values
        y_indices = np.where((y >= min_y) & (y <= max_y))[0]
        x_indices = np.where((x >= min_x) & (x <= max_x))[0]
        dataset_cut = dataset.isel(y=y_indices, x=x_indices)
        return dataset_cut

def run_pipeline(dir_path, data_name, ds_vars, pred_var=None):
    cleaned_uWRF_data = []
    cleaned_NAM_data = []

    if data_name == "uWRF":
        input_files = glob.glob(os.path.join(dir_path, '*'))
        for file in input_files:
            print(f"Processing file: {file}")
            loader = DataLoader(file, data_name, ds_vars, pred_var)
            dataset = loader.load_data()
            cleaner = uWRFDataCleaner(dataset)
            dataset = cleaner.rename_and_filter_vars(dataset, ds_vars, pred_var)
            cleaned_uWRF_data.append(dataset)
        
        final_dataset = xr.concat(cleaned_uWRF_data, dim='Time').sortby('Time')
        output_dir = '/D4/data/gvaillant/uwrf-cleaned'
        month = os.path.basename(os.path.normpath(dir_path))
        var_name = "WS" if set(ds_vars) == {"UGRD_10maboveground", "VGRD_10maboveground"} else "_".join(ds_vars)
        output_file = os.path.join(output_dir, f"uwrf_cleaned_{month}_{var_name}.nc")
        final_dataset.to_netcdf(output_file)
        print(f"Saved cleaned uWRF dataset to: {output_file}")

    elif data_name == "NAM":
        input_files = glob.glob(os.path.join(dir_path, '*'))
        for file in input_files:
            print(f"Processing file: {file}")
            loader = DataLoader(file, data_name, ds_vars, pred_var)
            dataset = loader.load_data()
            if dataset is None:
                continue
            cleaner = NAMDataCleaner(dataset)
            cutter = NAMSpatialCutter(dataset)
            dataset = cleaner.rename_and_filter_vars(dataset, ds_vars, pred_var)
            if dataset is not None:
                cleaned_NAM_data.append(dataset)

        if cleaned_NAM_data:
            final_dataset = xr.concat(cleaned_NAM_data, dim='time').sortby('time')
            final_dataset = cutter.cut(final_dataset)
            output_dir = '/D4/data/gvaillant/NAM-cleaned'
            month = os.path.basename(os.path.normpath(dir_path))
            var_name = "WS" if set(ds_vars) == {"UGRD_10maboveground", "VGRD_10maboveground"} else "_".join(ds_vars)
            output_file = os.path.join(output_dir, f"nam_cleaned_{month}_{var_name}.nc")
            final_dataset.to_netcdf(output_file)
            print(f"Saved cleaned NAM dataset to: {output_file}")
        else:
            print("No valid datasets found. Skipping concatenation.")
    else:
        raise ValueError("Invalid data name specified. Either uWRF or NAM.")

def main():
    parser = argparse.ArgumentParser(description="Run the data cleaning pipeline for uWRF or NAM data.")
    parser.add_argument("dir_path", type=str, help="Path to the directory containing input NetCDF files.")
    parser.add_argument("data_name", type=str, choices=["uWRF", "NAM"], help="Dataset name (uWRF or NAM).")
    parser.add_argument("ds_vars", nargs='+', type=str, help="Variable(s) to downscale.")
    parser.add_argument("--pred_var", type=str, default=None, help="Optional predictor variable.")
    args = parser.parse_args()
    run_pipeline(args.dir_path, args.data_name, args.ds_vars, args.pred_var)

if __name__ == "__main__":
    main()

