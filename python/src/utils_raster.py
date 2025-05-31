import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
from scipy import ndimage
import logging
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
    with rioxarray.open_rasterio(path, mask_and_scale=True) as raster:
            if "band" in raster.dims and raster.sizes["band"] == 1:
                raster = raster.squeeze("band", drop=True)
            raster = raster.rename(path.parent.stem)

            if scale:
                raster_max = raster.max().item()
                raster_min = raster.min().item()

                if (raster_max > 1) & (raster_min < 100):
                    logging.debug("Rescaling habitat quality between 0 and 1")
                    raster = raster / 100.  # Scale values
                else:
                    raise ValueError("Raster values are not in the expected range")
            
            # Return a copy of the raster to ensure the file handle can be closed
            return raster.copy()

def crop_raster(raster, buffer):
    buffered_gdf = gpd.GeoDataFrame(geometry=buffer)
    masked_raster = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    return masked_raster    
    

def dataset_to_geotiff(dataset, filepath):
    """
    Save a rioxarray dataset as a single multi-band GeoTIFF, 
    where each variable corresponds to a band.
    """
    stacked = dataset.to_dataarray(dim="band")
    stacked.attrs["band_names"] = list(dataset.data_vars)
    stacked.rio.to_raster(filepath, 
                          driver="GTiff", 
                          compress="zstd")
    
def load_geotiff_dataset(filepath):
    """
    Load a multi-band GeoTIFF where each band corresponds to a variable.
    Reconstruct the original dataset with variable names.
    """
    dataar = rioxarray.open_rasterio(filepath)
    dataset = dataar.to_dataset(dim="band")
    band_names = eval(dataar.attrs["band_names"])
    dataset = dataset.rename({i+1: n  for i, n in enumerate(band_names)})
    return dataset


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


if __name__ == "__main__":
    def mock_dataset():
        width = 100  # Number of grid cells in the x-direction
        height = 100  # Number of grid cells in the y-direction

        x_coords = np.linspace(0, 1000, width)
        y_coords = np.linspace(0, 1000, height)
        elevation = np.sin(np.linspace(0, 2 * np.pi, width)) * np.cos(np.linspace(0, 2 * np.pi, height))[:, None]
        temperature = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, width))[:, None] * np.cos(np.linspace(0, 2 * np.pi, height))

        ds = xr.Dataset(
            {
                "elevation": (("y", "x"), elevation),
                "temperature": (("y", "x"), temperature)
            },
            coords={"x": x_coords, "y": y_coords}
        )

        ds["elevation"].attrs["units"] = "meters"
        ds["elevation"].attrs["description"] = "Synthetic elevation data"
        ds["temperature"].attrs["units"] = "degrees Celsius"
        ds["temperature"].attrs["description"] = "Synthetic temperature data"

        ds = ds.rio.write_crs("EPSG:32633")  # Example: UTM Zone 33N
        return ds
    
    
    def test_dataset_to_geotiff():
        orig_dataset = mock_dataset()
        dataset_to_geotiff(orig_dataset, "synthetic_data.tif")
        dataset = load_geotiff_dataset("synthetic_data.tif")
        assert np.allclose(dataset["temperature"], orig_dataset["temperature"], atol=1e-6)