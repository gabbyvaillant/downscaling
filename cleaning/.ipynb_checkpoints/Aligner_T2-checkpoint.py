import xarray as xr
import numpy as np
import os

def regrid_nam_like_uwrf(nam_data, uwrf_data, target_shape=(30, 30)):

    nam_data = nam_data.rename({'TMP_2maboveground': 'T2'})
    nam_data = nam_data.rename({'PRES_surface':'PSFC'})

    # 1. Get uWRF lat/lon bounds from first time step
    uwrf_lat = uwrf_data.latitude.isel(Time=0).values
    uwrf_lon = uwrf_data.longitude.isel(Time=0).values
    lat_min, lat_max = uwrf_lat.min(), uwrf_lat.max()
    lon_min, lon_max = uwrf_lon.min(), uwrf_lon.max()

    # 2. Create target grid (lat/lon)
    new_lat = np.linspace(lat_min, lat_max, target_shape[0])
    new_lon = np.linspace(lon_min, lon_max, target_shape[1])
    lon2d, lat2d = np.meshgrid(new_lon, new_lat)

    # 3. Assign fake lat/lon to NAM's y and x to use for interpolation
    y_len = nam_data.dims['y']
    x_len = nam_data.dims['x']
    lat_guess = np.linspace(lat_min, lat_max, y_len)
    lon_guess = np.linspace(lon_min, lon_max, x_len)
    nam_data = nam_data.assign_coords({'y': lat_guess, 'x': lon_guess})

    # 4. Match time dimension length
    min_time = min(nam_data.sizes['time'], uwrf_data.sizes['Time'])
    nam_data = nam_data.isel(time=slice(0, min_time))

    # 5. Interpolate each variable
    interpolated_vars = {}
    for var in nam_data.data_vars:
        da = nam_data[var]
        # Interpolate using 'y' and 'x' which now represent lat/lon
        da_interp = da.interp(y=new_lat, x=new_lon, method='nearest')
        interpolated_vars[var] = (['time', 'y', 'x'], da_interp.values)

    # 6. Construct new Dataset with fixed lat/lon but keep x/y naming
    regridded_nam = xr.Dataset(
        data_vars=interpolated_vars,
        coords={
            'time': nam_data.time,
            'y': new_lat,
            'x': new_lon,
            'latitude': (('y', 'x'), lat2d),
            'longitude': (('y', 'x'), lon2d),
        },
        attrs=nam_data.attrs
    )

    return regridded_nam


    
def main():
    
    month = 5

    input_uwrf = xr.open_dataset(f'/D4/data/gvaillant/uwrf-cleaned/uwrf_cleaned_{str(month).zfill(2)}.nc')
    input_nam = xr.open_dataset(f'/D4/data/gvaillant/NAM-cleaned/nam_cleaned_{str(month).zfill(2)}_test.nc')
    output_dir = '/D4/data/gvaillant/NAM-aligned'

    aligned_ds = regrid_nam_like_uwrf(input_nam, input_uwrf)

    # Save to NetCDF
    output_path = os.path.join(output_dir, f'nam_aligned_{str(month).zfill(2)}.nc')
    aligned_ds.to_netcdf(output_path)
    print(f"Saved aligned NAM to {output_path}")

    
main()
