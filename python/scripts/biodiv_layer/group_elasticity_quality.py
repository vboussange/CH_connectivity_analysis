"""
Calculating the elasticity of habitat quality with respect to permeability using Jaxscape.
For simplification, we sum up qualities of all species within group and use the mean D for all species.
TODO: need to verify that the batching and calculation are correct.
"""
import jax
import numpy as np
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.gridgraph import GridGraph
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance

import equinox as eqx
from tqdm import tqdm
import sys
sys.path.append("./../../src")
from preprocessing import compile_quality
from processing import batch_run_calculation, padding
from TraitsCH import TraitsCH
from NSDM import NSDM
from EUSDM import EUSDM
import xarray as xr

from swissTLMRegio import MasksDataset, get_CH_border
from utils_raster import crop_raster, calculate_resolution, coarsen_raster, mask_raster
import pickle



def Kq(hab_qual, activities, distance, D):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=hab_qual,
                     nb_active=activities.size)

    window_center = jnp.array([[activities.shape[0]//2+1, activities.shape[1]//2+1]])
    
    dist = distance(grid, sources=window_center).reshape(-1)

    K = jnp.exp(-dist/D) # calculating proximity matrix
    
    epsilon = K * hab_qual[window_center[0, 0], window_center[0, 1]]
    epsilon = grid.node_values_to_array(epsilon)

    return epsilon


Kq_vmap = eqx.filter_vmap(Kq, in_axes=(0,0,None,None))

def compile_group_suitability(group, resolution):
    cache_path = Path(f".cache/group/{group}.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    traits = TraitsCH()
    species = traits.get_all_species_from_group(group)
    print(f"Group {config['group']} has {len(species)} species")
    
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

if __name__ == "__main__":
    
    config = {"group": "Reptiles",
              "batch_size": 18, # pixels, actual batch size is batch_size**2
              "resolution": 100, # meters
              "coarsening_factor": 9, # pixels, must be odd, where 1 is no coarsening
              "dtype": "float32",
             }

    distance = LCPDistance()
    mean_suitability, std_suitability, D_m = compile_group_suitability(config["group"], config["resolution"])

    output_path = Path("output") / config["group"]
    output_path.mkdir(parents=True, exist_ok=True)
    
