import os
import glob
import xarray as xr
import numpy as np
import argparse

# Define dataset-specific variables
NAM_VARS = [
    'TMP_2maboveground', 'PRES_surface', 'UGRD_10maboveground', 'VGRD_10maboveground', 
    'DSWRF_surface', 'HPBL_surface', 'HGT_surface'
]

uWRF_VARS = [
    'T2', 'PSFC', 'U10', 'V10', 'SWDOWN', 'PBLH', 'HGT'
]

class DataLoader:
    def __init__(self, dir_path, data_name):
        self.dir_path = dir_path
        self.data_name = data_name
        self.dataset = None

    def load_data(self):
        self.dataset = xr.open_dataset(self.dir_path)
        return self.dataset

class uWRFDataCleaner:
    def __init__(self, dataset, base_vars):
        self.dataset = dataset
        self.base_vars = base_vars

    def rename_and_filter_vars(self, dataset):
        vars_to_keep = [var for var in self.base_vars if var in dataset]
        dataset = dataset[vars_to_keep]
        dataset = dataset.rename({'XLAT': 'latitude', 'XLONG': 'longitude', 'XTIME': 'time'})
        return dataset

class NAMDataCleaner:
    def __init__(self, dataset, base_vars):
        self.dataset = dataset
        self.base_vars = base_vars
        self.rename_map = {
            'TMP_2maboveground': 'T2',
            'PRES_surface': 'PSFC',
            'UGRD_10maboveground': 'U10',
            'VGRD_10maboveground': 'V10',
            'DSWRF_surface': 'SWDOWN',
            'HPBL_surface': 'PBLH',
            'HGT_surface': 'HGT'
        }

    def rename_and_filter_vars(self, dataset):
        # Adjust longitudes to [-180, 180] range
        dataset['longitude'] = (dataset['longitude'] + 180) % 360 - 180

        # Keep only base_vars that are present
        vars_to_keep = [var for var in self.base_vars if var in dataset]
        missing_vars = [var for var in self.base_vars if var not in dataset]
        if missing_vars:
            print(f"Skipping dataset: Missing variables {missing_vars}")
            return None

        dataset = dataset[vars_to_keep]

        # Rename to match uWRF
        rename_subset = {k: v for k, v in self.rename_map.items() if k in dataset}
        dataset = dataset.rename(rename_subset)

        return dataset


class NAMSpatialCutter:
    def __init__(self, dataset):
        self.dataset = dataset

    def cut(self, dataset):
        return dataset.isel(y=slice(1, 41), x=slice(27, 67))
def run_pipeline(dir_path, data_name):
    cleaned_data = []
    base_vars = uWRF_VARS if data_name == "uWRF" else NAM_VARS

    input_files = glob.glob(os.path.join(dir_path, '*'))
    for file in input_files:
        print(f"Processing file: {file}")
        loader = DataLoader(file, data_name)
        dataset = loader.load_data()

        if data_name == "uWRF":
            cleaner = uWRFDataCleaner(dataset, base_vars)
            dataset = cleaner.rename_and_filter_vars(dataset)
        elif data_name == "NAM":
            cleaner = NAMDataCleaner(dataset, base_vars)
            cutter = NAMSpatialCutter(dataset)
            dataset = cleaner.rename_and_filter_vars(dataset)
        else:
            raise ValueError("Invalid dataset name. Use 'uWRF' or 'NAM'.")

        if dataset is not None:
            if data_name == "NAM":
                dataset = cutter.cut(dataset)
            cleaned_data.append(dataset)

    if cleaned_data:
        concat_dim = 'Time' if data_name == 'uWRF' else 'time'
        final_dataset = xr.concat(cleaned_data, dim=concat_dim).sortby(concat_dim)
        output_dir = f"/D4/data/gvaillant/{data_name}-cleaned"
        os.makedirs(output_dir, exist_ok=True)
        month = os.path.basename(os.path.normpath(dir_path))
        output_file = os.path.join(output_dir, f"{data_name.lower()}_cleaned_{month}.nc")
        final_dataset.to_netcdf(output_file)
        print(f"Saved cleaned {data_name} dataset to: {output_file}")
    else:
        print("No valid datasets found. Skipping concatenation.")

def main():
    parser = argparse.ArgumentParser(description="Run the data cleaning pipeline for uWRF or NAM data.")
    parser.add_argument("dir_path", type=str, help="Path to the directory containing input NetCDF files.")
    parser.add_argument("data_name", type=str, choices=["uWRF", "NAM"], help="Dataset name (uWRF or NAM).")
    args = parser.parse_args()
    run_pipeline(args.dir_path, args.data_name)

if __name__ == "__main__":
    main()

