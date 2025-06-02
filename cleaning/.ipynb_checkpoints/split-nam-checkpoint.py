import xarray as xr
import os

def main():
    file_list = [
        '/D4/data/gvaillant/NAM-aligned/nam_aligned_01.nc',
        '/D4/data/gvaillant/NAM-aligned/nam_aligned_02.nc',
        '/D4/data/gvaillant/NAM-aligned/nam_aligned_03.nc',
        '/D4/data/gvaillant/NAM-aligned/nam_aligned_04.nc',
        '/D4/data/gvaillant/NAM-aligned/nam_aligned_05.nc',
    ]

    datasets = [xr.open_dataset(file) for file in file_list]

    combined_ds = xr.concat(datasets, dim='time')
    combined_ds = combined_ds.sortby('time')

    # Split
    num_time_steps = combined_ds.sizes['time']
    train_end = int(0.7 * num_time_steps)
    val_end = int(0.85 * num_time_steps)

    train_ds = combined_ds.isel(time=slice(0, train_end))
    val_ds = combined_ds.isel(time=slice(train_end, val_end))
    test_ds = combined_ds.isel(time=slice(val_end, num_time_steps))

    # Save directory
    save_dir = '/D4/data/gvaillant/nam-split'
    os.makedirs(save_dir, exist_ok=True)

    # Save datasets
    train_ds.to_netcdf(os.path.join(save_dir, 'nam_train.nc'))
    val_ds.to_netcdf(os.path.join(save_dir, 'nam_val.nc'))
    test_ds.to_netcdf(os.path.join(save_dir, 'nam_test.nc'))

    print("Files saved successfully.")

main()
