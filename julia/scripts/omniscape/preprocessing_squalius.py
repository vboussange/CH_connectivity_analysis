"""
Computing genetic vs ecological vs euclidean distance.
This script need to be checked.
"""

import pyreadr
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd
import numpy as np
import jax.numpy as jnp
import sys
sys.path.append("./../../../python/src")
# from swissTLMRegio import MasksDataset, get_canton_border
from utils_raster import load_raster, NSDM25m_PATH, CRS_CH, calculate_resolution
from TraitsCH import TraitsCH
import geopandas as gpd
import pandas as pd



if __name__ == "__main__":
    # loading source raster
    raster_path = Path("inputs/source.tif")
    source_raster = load_raster(raster_path)
    # raster = raster.rio.reproject(CRS_CH)
    
    # creating resistance
    resistance_raster = 1 / source_raster
    resistance_raster.rio.to_raster("inputs/resistance.tif")
    
    
    output_file = Path("output/test_rupicapra_100m")
    buffer_distance = 0 # meters
    resolution = 100 # meters
    traits = TraitsCH()
    species = "Rupicapra rupicapra"
    dispersal_distance = traits.get_D(species)
    guilds = traits.get_guilds(species)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    
    print("Original raster of resolution:")
    lat_resolution, lon_resolution = calculate_resolution(raster)
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
    