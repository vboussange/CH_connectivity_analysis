import xarray as xr
import rioxarray as rxr
import netCDF4
import numpy as np

# Open a geospatial dataset using rioxarray
file_name = "../../../data/GUILDS_EU_SP/GUILDS_EU_SP_Zug_resampling_1.nc"
dataset = xr.open_dataset(file_name, engine='netcdf4', decode_coords="all")

# Parameters for the rolling window and buffer
window_size = 5  # e.g., a 5x5 window
buffer_size = 2  # buffer size on either side

# Apply the rolling window along both dimensions
rolling_dataset = dataset.rolling(
    dim={'x': window_size + 2 * buffer_size, 'y': window_size + 2 * buffer_size},
    center=True
).construct({'x': 'window_x', 'y': 'window_y'})

# Extract the core moving window (without the buffer)
window = rolling_dataset.isel(window_x=slice(buffer_size, -buffer_size),
                              window_y=slice(buffer_size, -buffer_size))

# Apply a function to each window (e.g., mean or custom function)
result = window.reduce(np.mean, dim=('window_x', 'window_y'))

# Saving the result back to a geospatial file
result.rio.to_raster("path_to_output_file.tif")
