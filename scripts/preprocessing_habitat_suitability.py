"""
Cropping habitat suitability rasters with a buffer around CH, and upsampling.
"""

import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path

SWITZERLAND_BOUNDARY_PATH = Path(__file__).parent / '../../data/swiss_boundaries/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp'

def coarsen_raster(raster, resampling_factor):
    print("Original raster of resolution")
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    print(f"Latitude resolution: {lat_resolution}")
    print(f"Longitude resolution: {lon_resolution}")
    
    
    # calculating new resolution
    # Assuming 'data' is your xarray DataArray
    raster = raster.coarsen(x=resampling_factor, y=resampling_factor, boundary='trim').mean()
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    print("Coarse raster of resolution")
    print(f"Latitude resolution: {lat_resolution}")
    print(f"Longitude resolution: {lon_resolution}")
    return raster
    
def load_raster(path):
    # Load the raster file
    raster = rioxarray.open_rasterio(path, mask_and_scale=True)
    raster = raster.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
    raster = raster.rename(path.stem)
    return raster

def crop_raster(raster):
    switzerland_boundary = gpd.read_file(SWITZERLAND_BOUNDARY_PATH)
    switzerland_buffer = switzerland_boundary.buffer(buffer_distance)

    if switzerland_buffer.crs != raster.rio.crs:
        raster = raster.rio.reproject(switzerland_buffer.crs)
    buffered_gdf = gpd.GeoDataFrame(geometry=switzerland_buffer)
    masked_raster = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    return masked_raster



if __name__ == "__main__":
    buffer_distance = 50000  # 100km in meters
    resampling_factor = 4
    
    input_dir = Path(__file__).parent / '../../data/GUILDES_EU/'
    output_file = input_dir.parent / f"{input_dir.stem}_buffer_dist={int(buffer_distance/1000)}km_resampling_{resampling_factor}.nc"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    raster_files = list(Path(input_dir).glob('*.tif'))
    rasters = [load_raster(file) for file in raster_files]
    cropped_rasters = [crop_raster(raster) for raster in rasters]

    cropped_and_coarsened_raster = [coarsen_raster(raster, resampling_factor) for raster in cropped_rasters]
    
    
    print(f"saved at {output_file}")
    dataset = xr.merge(cropped_and_coarsened_raster, join="left")
    dataset.to_netcdf(output_file, engine='netcdf4')
    
    dataset = rioxarray.open_rasterio(output_file, mask_and_scale=True)