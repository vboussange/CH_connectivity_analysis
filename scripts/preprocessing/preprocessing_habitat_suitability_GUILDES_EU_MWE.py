"""
Cropping habitat suitability rasters with a buffer around CH, and upsampling.
Saving all guilds in an xarrray dataset within a netcdf.

# TODO: you may consider for aquatic bodies to use connection points
"""

import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd
from swissTLMRegio import MasksDataset, get_canton_border
from TraitsCH import TraitsCH

CRS = "EPSG:2056" # https://epsg.io/2056
                
def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution

def coarsen_raster(raster, resampling_factor):
    raster_coarse = raster.coarsen(x=resampling_factor, y=resampling_factor, boundary='trim').mean()
    raster_coarse.rio.set_crs(raster.rio.crs)
    return raster_coarse
    
def load_raster(path):
    # Load the raster file
    raster = rioxarray.open_rasterio(path, mask_and_scale=True)
    raster = raster.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
    raster = raster.rename(path.parent.stem)
    return raster

def crop_raster(raster, buffer):
    buffered_gdf = gpd.GeoDataFrame(geometry=buffer)
    masked_raster = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    return masked_raster


def mask_raster(raster, traits_dataset, masks_dataset):
    sp_name = raster.name
    hab = traits_dataset.get_habitat(sp_name)
    if hab == "Aqu":
        mask = masks_dataset[hab]
        raster_masked = raster.rio.clip(mask, all_touched=True, drop=True)
        
    else:
        raster_masked = raster
    return raster_masked



if __name__ == "__main__":
    buffer_distance = 500 # meters
    resampling_factor = 1
    canton = "Zug"
    boundary = get_canton_border("Zug")
    boundary_buffer = boundary.buffer(buffer_distance)

    input_dir = Path(__file__).parent / '../../../data/GUILDS_EU_SP/'
    output_file = input_dir / f"{input_dir.stem}_{canton}_resampling_{resampling_factor}.nc"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    masks_dataset = MasksDataset()
    traits_dataset = TraitsCH()

    # raster_files = list(Path(input_dir).glob('**/*.tif'))
    raster_files = [Path("/Users/victorboussange/projects/connectivity/connectivity_analysis/code/scripts/preprocessing/../../../data/GUILDS_EU_SP/Salmo trutta/Salmo.trutta_glo_ensemble.tif")]
    rasters = [load_raster(file) for file in raster_files]
    # reprojections
    rasters = [rast.rio.reproject(CRS) for rast in rasters]
    
    # masking
    rasters = [mask_raster(rast, traits_dataset, masks_dataset) for rast in rasters]

    print("Original raster of resolution:")
    lat_resolution, lon_resolution = calculate_resolution(rasters[0])
    print(f"Latitude resolution: {lat_resolution/1000:0.2f}km")
    print(f"Longitude resolution: {lon_resolution/1000:0.2f}km")
    
    cropped_rasters = [crop_raster(raster, boundary_buffer) for raster in rasters]

    cropped_and_coarsened_raster = [coarsen_raster(raster, resampling_factor) for raster in cropped_rasters]
    print("Coarse raster of resolution:")
    lat_resolution, lon_resolution = calculate_resolution(cropped_and_coarsened_raster[0])
    print(f"Latitude resolution: {lat_resolution/1000:0.2f}km")
    print(f"Longitude resolution: {lon_resolution/1000:0.2f}km")
    
    print(f"saved at {output_file}")
    dataset = xr.merge(cropped_and_coarsened_raster, join="left")
    dataset.to_netcdf(output_file, engine='netcdf4')
    
    # dataset = xr.open_dataset(output_file, engine='netcdf4', decode_coords="all")