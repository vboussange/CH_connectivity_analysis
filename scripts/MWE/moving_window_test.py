import xarray as xr
import rioxarray
import numpy as np

path = "../../../data/GUILDS_EU_SP/GUILDS_EU_SP_buffer_dist=100km_resampling_1.nc"
dataset = xr.open_dataset(path, engine="netcdf4", decode_coords="all")   

sp_name = "Salmo trutta"

# width and height of window center
window_size = 40
buffer_size = 3
step_size = window_size  # Step size for non-overlapping core windows
cut_off = 0.1

# calculate rolling window
total_window_size = window_size + 2 * buffer_size

data_array = dataset[sp_name] / 100

output_array = xr.full_like(data_array, np.nan)


x_start = 10
y_start=10
output_array.values[0,
                    (x_start + buffer_size):(x_start + buffer_size + window_size),
                    (y_start + buffer_size):(y_start + buffer_size + window_size)] = 1

# data_array.values[buffer_size:buffer_size+window_size, buffer_size:buffer_size+window_size]


