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
from swissTLMRegio import MasksDataset, get_CH_border
from utils_raster import crop_raster, calculate_resolution, coarsen_raster, mask_raster
from TraitsCH import TraitsCH
from NSDM import NSDM
from EUSDM import EUSDM
import jax.numpy as jnp
import pickle 

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


def compile_group_suitability(group, resolution):
    cache_path = Path(f".cache/group/{group}.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    traits = TraitsCH()
    species = traits.get_all_species_from_group(group)
    if len(species) == 0:
        raise ValueError(f"No data found for group {group}")
    print(f"Group {group} has {len(species)} species")
    
    D_m = np.mean([traits.get_D(sp) for sp in species]) * 1000 # convert to meters
    switzerland_boundary = get_CH_border()
    switzerland_buffer = switzerland_boundary.buffer(D_m)

    nsdm_dataset = NSDM()
    eusdm_dataset = EUSDM()
    nsdm_rasters = []
    eu_sdm_rasters = []
    for sp in species:
        try:
            nsdm_rasters.append(nsdm_dataset.load_raster(sp))
            raster_coarse = eusdm_dataset.load_raster(sp)
            raster_coarse = crop_raster(raster_coarse.squeeze(), switzerland_buffer)
            eu_sdm_rasters.append(raster_coarse)
        except:
            continue
            # print(f"Failed to load raster for {sp}")
    print(f"Loaded {len(nsdm_rasters)} rasters")
    # 
    nsdm_dataset = xr.concat(nsdm_rasters, dim="species")
    mean_ch_sdm_suitability = nsdm_dataset.mean(dim="species").squeeze("band").rename("mean_nsdm")    
    std_suitability = nsdm_dataset.std(dim="species").squeeze("band").rename("std_nsdm")    

    
    eu_sdm_dataset = xr.concat(eu_sdm_rasters, dim="species")
    mean_eu_sdm_suitability = eu_sdm_dataset.mean(dim="species").rename("mean_eu_sdm") 
    
    raster_coarse_interp = mean_eu_sdm_suitability.interp_like(mean_ch_sdm_suitability, method="nearest")
    combined_raster = xr.where(mean_ch_sdm_suitability.notnull(), mean_ch_sdm_suitability, raster_coarse_interp)
    combined_raster = combined_raster.rename(species.iloc[0]) # TODO: maybe fix, needed for masking
    combined_raster.rio.set_crs(nsdm_rasters[0].rio.crs, inplace=True)

    # masking out land for aquatic species
    combined_raster = mask_raster(combined_raster, traits, MasksDataset())
    combined_raster.rio.set_crs(nsdm_rasters[0].rio.crs, inplace=True)

    # resampling raster
    lat_resolution, lon_resolution = calculate_resolution(combined_raster)
    print(f"Original quality resolution: {lon_resolution/1000:0.3f}km")
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    mean_suitability = coarsen_raster(combined_raster, resampling_factor)
    mean_suitability = mean_suitability.rename(group)
    with open(cache_path, "wb") as f:
        pickle.dump((mean_suitability, std_suitability, D_m), f)
    
    return mean_suitability, std_suitability, D_m

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