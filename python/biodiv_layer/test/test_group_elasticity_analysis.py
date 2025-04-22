"""
Calculating the elasticity of habitat quality with respect to permeability for
all taxonomic groups in `jaxscape`.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use the first GPU
import jax
import math
import sys
import xarray as xr
import rioxarray
import jax.numpy as jnp
from pathlib import Path
from jaxscape import LCPDistance, GridGraph
from jaxscape.sensitivity_analysis import SensitivityAnalysis, d_permeability_vmap
from copy import deepcopy
import git
from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(str(Path(__file__).parent / Path("../../src/")))
from group_preprocessing import compile_group_suitability, CRS_CH, GROUP_INFO
from utils_raster import upscale, downscale, crop_raster, calculate_resolution
from masks import get_CH_border
import numpy as np
import xarray as xr

def create_synthetic_suitability_dataset():
    
    grid_points = np.arange(0, 10000, config["resolution"])
    mean_suitability_values = np.random.uniform(0, 0.2, (grid_points.size, grid_points.size))
    mean_suitability_values[10:150, 10:150] = 0.8
    mean_suitability_values[200:300, 200:300] = 0.8

    suitability_dataset = xr.Dataset(
        {
            "mean_suitability": (("y", "x"), mean_suitability_values)
        },
        coords={
            "y": grid_points,
            "x": grid_points
        }
    )
    suitability_dataset.rio.write_crs(CRS_CH, inplace=True)
    suitability_dataset.attrs["D_m"] = 2000  # Example dispersal range
    dependency_range = math.ceil(3 * suitability_dataset.attrs["D_m"] / config["resolution"])
    suitability_dataset_padded = jnp.pad(
        suitability_dataset,
        ((dependency_range, dependency_range),
        (dependency_range, dependency_range)),
        mode="constant",
        constant_values=0
    )
    return suitability_dataset_padded

def proximity(dist, D_m, alpha):
    return jnp.exp(-dist * alpha / D_m)

# def proximity(dist, D_m, alpha):
#     return jnp.exp(-dist) / jnp.sum(jnp.exp(-dist))

def run_elasticity_analysis_for_group(config):
    """
    Runs elasticity analysis for a single group using the given configuration.
    """
    distance_fn = LCPDistance()
    
    suitability_dataset = create_synthetic_suitability_dataset()
    fine_resolution, _ = calculate_resolution(suitability_dataset["mean_suitability"])
    D_m = suitability_dataset.attrs["D_m"] # mean dispersal range

    upscale_resolution = max(fine_resolution, D_m * config["analysis_precision"])
    quality_raster = upscale(suitability_dataset["mean_suitability"], upscale_resolution)

    quality = jnp.array(quality_raster.values, dtype=config["dtype"])
    quality = jnp.nan_to_num(quality, nan=0.0)
    quality = jnp.where(quality == 0, 1e-5, quality)
    
    dependency_range = math.ceil(3 * D_m / upscale_resolution)
    mean_dist = 1 / jnp.mean(quality)
    alpha = upscale_resolution / mean_dist
    
    plt.imshow(quality)
    plt.title("Coarse scale suitability")
    plt.show()

    sensitivity_analyzer = SensitivityAnalysis(
        quality_raster=quality,
        permeability_raster=quality,
        distance=distance_fn,
        proximity=lambda dist: proximity(dist, D_m, alpha),
        coarsening_factor=0.,
        dependency_range=dependency_range,
        batch_size=10
    )

    output = sensitivity_analyzer.run("permeability")[dependency_range:-dependency_range, dependency_range:-dependency_range]
    plt.imshow(output)
    plt.title("Raw output")

    output_raster = deepcopy(quality_raster)
    output_raster.rio.set_crs(CRS_CH, inplace=True)
    output_raster.values = output
    output_raster = downscale(output_raster, suitability_dataset["mean_suitability"])
    elasticity_raster = output_raster * suitability_dataset["mean_suitability"]
    plt.imshow(elasticity_raster)
    plt.colorbar()
    plt.title("Elasticity raster")
    plt.show()

if __name__ == "__main__":
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)

    config = {
        "batch_size": 32,
        "dtype": "float32",
        "analysis_precision": 2e-1, # percentage of the dispersal range
        "resolution": 25,            # meters
        "hash": sha
    }
    run_elasticity_analysis_for_group(config)