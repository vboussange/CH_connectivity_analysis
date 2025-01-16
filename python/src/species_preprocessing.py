"""
Generating quality and permeability maps for a terrestrial species in CH
"""
# import pyproj
# required due to multiple pyproj installations
# pyproj.datadir.set_data_dir("/Users/victorboussange/projects/connectivity/connectivity_analysis/code/python/.env/share/proj/")
import xarray as xr
from shapely.geometry import box
from pathlib import Path
import numpy as np
from masks import MasksDataset, get_CH_border
from utils_raster import crop_raster, calculate_resolution, coarsen_raster, mask_raster, CRS_CH, fill_na_with_nearest
from TraitsCH import TraitsCH
from NSDM import NSDM
from EUSDM import EUSDM
from tqdm import tqdm

from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance

def compile_species_suitability(species_name, D_m, resolution):
    # loading fine resolution raster
    raster_NSDM = NSDM().load_raster(species_name)
    lat_resolution, lon_resolution = calculate_resolution(raster_NSDM)
    print(f"Original quality resolution: {lon_resolution/1000:0.3f}km")
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    raster_NSDM = coarsen_raster(raster_NSDM, resampling_factor)
    # raster_NSDM = raster_NSDM.fillna(0)
    
    # loading coarse resolution raster
    try:
        raster_coarse = EUSDM().load_raster(species_name)
        switzerland_boundary = get_CH_border()
        switzerland_buffer = switzerland_boundary.buffer(D_m)
        raster_coarse = crop_raster(raster_coarse.squeeze(), switzerland_buffer)
        
        # merging coarse and fine rasters 
        raster_coarse_interp = raster_coarse.interp_like(raster_NSDM, method="nearest")
        combined_raster = xr.where(raster_NSDM.notnull(), raster_NSDM, raster_coarse_interp)
        combined_raster.rio.set_crs(raster_NSDM.rio.crs, inplace=True)
        combined_raster = combined_raster.rename(species_name)
        
    except Exception as e:
        print(f"Failed to load coarse resolution raster: {e}")
        combined_raster = raster_NSDM
    
    # masking out land for aquatic species
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