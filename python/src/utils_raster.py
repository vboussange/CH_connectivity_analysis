import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
from scipy import ndimage
CRS_CH = "EPSG:2056" # https://epsg.io/2056

def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution

def coarsen_raster(raster, resampling_factor):
    raster_coarse = raster.coarsen(x=resampling_factor, y=resampling_factor, boundary='trim').mean()
    raster_coarse.rio.set_crs(raster.rio.crs, inplace=True)
    return raster_coarse

def upscale(raster, resolution):
    lat_resolution, lon_resolution = calculate_resolution(raster)
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    raster = coarsen_raster(raster, resampling_factor)
    return raster

def downscale(raster, ref_raster):
    raster = raster.interp_like(ref_raster, method="nearest")
    raster.rio.set_crs(ref_raster.rio.crs, inplace=True)
    return raster
    
def load_raster(path, scale=True):
    # Load the raster file
    raster = rioxarray.open_rasterio(path, mask_and_scale=True)
    raster = raster.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
    raster = raster.rename(path.parent.stem)
    if scale:
        if (raster.max() > 1) & (raster.min() < 100):
            print("Rescaling habitat quality between 0 and 1")
            raster = raster / 100.
        else:
            raise ValueError("raster values are not in the expected range")
    return raster

def crop_raster(raster, buffer):
    buffered_gdf = gpd.GeoDataFrame(geometry=buffer)
    masked_raster = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    return masked_raster


def mask_raster(raster, traits_dataset, masks_dataset):
    sp_name = raster.name
    hab = traits_dataset.get_habitat(sp_name)
    if hab in masks_dataset.masks.keys():
        mask = masks_dataset[hab]
        return raster.rio.clip(mask, all_touched=True, drop=True)
    
    else:
        return raster


def save_to_netcdf(dataset, path, scale_factor):
    # TODO: to test
    encoding = {}

    for var_name in dataset.data_vars:
        # Add compression settings for the variable
        encoding[var_name] = {
            "zlib": True,          # Enable compression
            "complevel": 5,        # Compression level (1–9)
            "dtype": "int16",      # Specify data type
            "scale_factor": scale_factor,  # Add scale factor for metadata
            "add_offset": 0,       # Add offset for metadata
        }

    # Save the dataset to a NetCDF file with the specified encoding
    dataset.to_netcdf(path, encoding=encoding)
    
def fill_na_with_nearest(da: xr.DataArray) -> xr.DataArray:
    """
    Fill NA values by assigning each NA cell the value of its nearest non-NA cell
    in Euclidean distance, preserving original non-NA cells.
    """
    data_np = da.values
    mask = np.isnan(data_np)
    
    # distance_transform_edt gives us for each point in `mask`:
    #   - the distance to the nearest False (i.e., nearest valid)
    #   - the indices of that nearest valid
    # So first invert mask to mark non-NA as False, NA as True:
    dist, (inds_y, inds_x, *other_inds) = ndimage.distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True
    )
    # NOTE: If da has more than 2 dimensions, you get multiple index arrays.
    # E.g. for 3D, you'll get (inds_z, inds_y, inds_x).
    # We'll just keep a conceptual example for 2D or shape out as needed.
    
    filled_np = data_np.copy()
    # Fill the NA cells using the nearest valid cell’s value
    filled_np[mask] = data_np[inds_y[mask], inds_x[mask]]
    
    # Return a new DataArray with the same coords, etc.
    filled_da = da.copy()
    filled_da.values = filled_np
    return filled_da