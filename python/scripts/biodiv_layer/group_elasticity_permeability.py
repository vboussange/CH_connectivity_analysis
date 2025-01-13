"""
Calculating the elasticity of habitat quality with respect to permeability for
all taxonomic group considered using `jaxscape`.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use the first GPU
# Change working directory to the directory of the file
import jax
import numpy as np
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance
from jaxscape.sensitivity_analysis import SensitivityAnalysis, d_permeability_vmap
import math

import sys
sys.path.append(str(Path(__file__).parent / Path("../../src/")))
from preprocessing import compile_group_suitability, CRS_CH
from processing import batch_run_calculation, padding, GROUP_INFO
from utils_raster import upscale, downscale, crop_raster, calculate_resolution
from masks import get_CH_border
import xarray as xr
import rioxarray
from copy import deepcopy
os.chdir(Path(__file__).parent)

def proximity(dist):
    return jnp.exp(-dist) / jnp.sum(jnp.exp(-dist))

if __name__ == "__main__":
    
    config = {"batch_size": 16, # pixels, actual batch size is batch_size**2
            "dtype": "float32",
            "analysis_precision": 5e-2, # percentage of the dispersal range defining the resolution of the analysis, must be less than 1
            "resolution": 25 # meters, resolution of the input/output raster
            } 

    for group in GROUP_INFO:
        print("Computing elasticity for group:", group)
        distance = GROUP_INFO[group]
        try:
            if not type(distance) is EuclideanDistance:
                output_path = Path("output") / group
                output_path.mkdir(parents=True, exist_ok=True)
                
                suitability_dataset = compile_group_suitability(group, config["resolution"])
                fine_resolution, _ = calculate_resolution(suitability_dataset["mean_suitability"])
                D_m = suitability_dataset.attrs["D_m"]
                
                # upscale raster
                upscale_resolution = max(fine_resolution, D_m * config["analysis_precision"])
                quality_raster = upscale(suitability_dataset["mean_suitability"], upscale_resolution)
                
                # to jax array  
                quality = jnp.array(quality_raster.values, dtype=config["dtype"])
                quality = jnp.nan_to_num(quality, nan=0.0)
                quality = jnp.where(quality == 0, 1e-5, quality)
                assert jnp.all(quality > 0) and jnp.all(quality < 1) and jnp.all(jnp.isfinite(quality)), "Quality values must be between 0 and 1."
                
                # dependency range corresponds to dispersal in pixels
                dependency_range = math.ceil(D_m / upscale_resolution)
                assert dependency_range >= 1, "Dispersal range must be greater than 1 pixel."
                                
                prob = SensitivityAnalysis(quality_raster=quality,
                                            permeability_raster=quality,
                                            distance=distance,
                                            proximity=proximity,
                                            coarsening_factor=0.,
                                            dependency_range=dependency_range,
                                            batch_size=config["batch_size"])
                # for quality, use d_quality_vmap
                output = prob.run(d_permeability_vmap)
                
                # transform output to raster
                # TODO: tif and nc files not consistent
                output_raster = deepcopy(quality_raster)
                output_raster.rio.set_crs(CRS_CH, inplace=True)
                output_raster.values = output
                switzerland_boundary = get_CH_border()
                
                # downscale raster to original resolution
                output_raster = downscale(output_raster, suitability_dataset["mean_suitability"])
                output_raster = crop_raster(output_raster, switzerland_boundary)
                elasticity_raster = output_raster * suitability_dataset["mean_suitability"]
                elasticity_raster.rio.to_raster(output_path / "elasticity_permeability.tif", compress='lzw')
                print("Saved elasticity raster at:", output_path / "elasticity_permeability.tif")
        
        except Exception as e:
            print(f"Failed to compute elasticity for group {group}: {e}")
            continue
