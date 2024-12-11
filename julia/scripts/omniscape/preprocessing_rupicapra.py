"""
Generating quality and permeability maps for a terrestrial species in CH
"""
import pyproj
# required due to multiple pyproj installations
pyproj.datadir.set_data_dir("/Users/victorboussange/projects/connectivity/connectivity_analysis/code/python/.env/share/proj/")
from pyproj import CRS
# pyproj.datadir.get_data_dir()
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
from swissTLMRegio import MasksDataset, get_CH_border, DamsDataset, WKADataset
from utils_raster import crop_raster, calculate_resolution, coarsen_raster, mask_raster
from TraitsCH import TraitsCH
import geopandas as gpd
import pandas as pd
from NSDM import NSDM
from EUSDM import EUSDM
import json

def compile_quality(species_name, D_m, resolution):
    # loading fine resolution raster
    raster_NSDM = NSDM().load_raster(species_name)
    lat_resolution, lon_resolution = calculate_resolution(raster_NSDM)
    print(f"Resolution: {lon_resolution/1000:0.3f}km")
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    raster_NSDM = coarsen_raster(raster_NSDM, resampling_factor)
    # raster_NSDM = raster_NSDM.fillna(0)
    
    # loading coarse resolution raster
    raster_coarse = EUSDM().load_raster(species_name)
    switzerland_boundary = get_CH_border()
    switzerland_buffer = switzerland_boundary.buffer(D_m)
    raster_coarse = crop_raster(raster_coarse.squeeze(), switzerland_buffer)

    # merging coarse and fine rasters 
    raster_coarse_interp = raster_coarse.interp_like(raster_NSDM, method="nearest")
    combined_raster = xr.where(raster_NSDM.notnull(), raster_NSDM, raster_coarse_interp)
    combined_raster.rio.set_crs(raster_NSDM.rio.crs)
    
    
    # masking out habitat
    combined_raster = mask_raster(combined_raster, TraitsCH(), MasksDataset())
    return combined_raster

# TODO: fix epsilon
def compile_resistance(quality, barriers, eps =  1e-5):
    
    # initial resistance is uniquely based on quality
    initial_resistance = - np.log(quality + eps) 
    
    # adding extra resistances due to barriers
    final_resistance = initial_resistance.copy()
    for barrier in barriers:
        extra_resistances = xr.full_like(quality, fill_value=-np.log(eps))
        extra_resistances.rio.set_crs(quality.rio.crs)
        barrier = barrier.get_mask()
        extra_resistances = extra_resistances.rio.clip(barrier, all_touched=True, drop=False)
        
        final_resistance = xr.where(extra_resistances.notnull(), extra_resistances, final_resistance)
        # we filter out out of domain values
        final_resistance = xr.where(initial_resistance.notnull(), final_resistance, initial_resistance)
    
    return final_resistance.rio.set_crs(quality.rio.crs)


if __name__ == "__main__":
    species_name = "Rupicapra rupicapra"
    output_path = Path("output") / species_name
    output_path.mkdir(parents=True, exist_ok=True)
    resolution = 100 # meters
    traits = TraitsCH()
    D_m = traits.get_D(species_name) * 1000 # in meters

    quality = compile_quality(species_name, D_m, resolution)
    resistance = compile_resistance(quality, [])
    
    resistance = resistance.rio.write_nodata(np.nan, encoded=True)
    quality = quality.rio.write_nodata(np.nan, encoded=True)
    resistance.rio.to_raster(output_path / "resistance.tif", dtype="float32")
    quality.rio.to_raster(output_path / "quality.tif", dtype="float32")
    
    # Save info in a json file
    D_m_data = {"species_name": species_name, 
                "D_m": D_m,
                "resolution": resolution}
    with open(output_path / "info.json", "w") as json_file:
        json.dump(D_m_data, json_file)
    