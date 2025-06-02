import xarray as xr
import os

def main():
    file_list = [
        '/D4/data/gvaillant/uwrf-cleaned/WS/uwrf_cleaned_01_U10_V10.nc',
        '/D4/data/gvaillant/uwrf-cleaned/WS/uwrf_cleaned_02_U10_V10.nc',
        '/D4/data/gvaillant/uwrf-cleaned/WS/uwrf_cleaned_03_U10_V10.nc',
        '/D4/data/gvaillant/uwrf-cleaned/WS/uwrf_cleaned_04_U10_V10.nc',
        '/D4/data/gvaillant/uwrf-cleaned/WS/uwrf_cleaned_05_U10_V10.nc',
    ]

    fixed_files = []

    for i in file_list:
        file = xr.open_dataset(i)
        file = file.rename({'Time': 'time'})
        if 'time' not in file.coords and 'time' in file:
            file = file.set_index(time='time')
        file = file.swap_dims({'time': 'time'})
        file = file.rename({'south_north': 'y', 'west_east': 'x'})
        fixed_files.append(file)

    combined_ds = xr.concat(fixed_files, dim='time')
    combined_ds = combined_ds.sortby('time')

    # Split
    num_time_steps = combined_ds.sizes['time']
    train_end = int(0.7 * num_time_steps)
    val_end = int(0.85 * num_time_steps)

    train_ds = combined_ds.isel(time=slice(0, train_end))
    val_ds = combined_ds.isel(time=slice(train_end, val_end))
    test_ds = combined_ds.isel(time=slice(val_end, num_time_steps))

    # Save directory
    save_dir = '/D4/data/gvaillant/uwrf-split'
    os.makedirs(save_dir, exist_ok=True)

    # Save datasets
    train_ds.to_netcdf(os.path.join(save_dir, 'uwrf_train_WS.nc'))
    val_ds.to_netcdf(os.path.join(save_dir, 'uwrf_val_WS.nc'))
    test_ds.to_netcdf(os.path.join(save_dir, 'uwrf_test_WS.nc'))

    print("Files saved successfully.")

main()
