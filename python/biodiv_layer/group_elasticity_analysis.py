"""
Calculating the elasticity of ecological landscape with respect to permeability or quality for
all taxonomic groups using `jaxscape`.
"""
import os
import argparse

parser = argparse.ArgumentParser(description='Run elasticity analysis.')
parser.add_argument('--group', default='Mammals', help='Taxonomic group')
parser.add_argument('--hab', default='Ter', help='Habitat type (Aqu/Ter)')
parser.add_argument('--gpu_id', default='0', help='GPU ID to use')
args = parser.parse_args()

# Set GPU before importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import jax
import math
import sys
import xarray as xr
import rioxarray
import jax.numpy as jnp
from pathlib import Path
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.sensitivity_analysis import SensitivityAnalysis, d_permeability_vmap
from copy import deepcopy
import git
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(str(Path(__file__).parent / Path("../src/")))
from group_preprocessing import compile_group_suitability, CRS_CH, GROUP_INFO
from utils_raster import upscale, downscale, crop_raster, calculate_resolution
from masks import get_CH_border

def proximity(dist, D_m, alpha):
    return jnp.exp(-dist * alpha / D_m)

def run_elasticity_analysis_for_group(group, hab, sens_type, config):
    """
    Runs elasticity analysis for a single group using the given configuration.
    """
    print(f"Running calculation for group {group}, habitat {hab}, sensitivity {sens_type}")
    
    distance_fn = GROUP_INFO[group]
    if isinstance(distance_fn, EuclideanDistance) and sens_type == "permeability":
        return

    suitability_dataset = compile_group_suitability(group, hab, config["resolution"])
    fine_resolution, _ = calculate_resolution(suitability_dataset["mean_suitability"])
    D_m = suitability_dataset.attrs["D_m"]

    upscale_resolution = max(fine_resolution, D_m * config["analysis_precision"])
    quality_raster = upscale(suitability_dataset["mean_suitability"], upscale_resolution)

    quality = jnp.array(quality_raster.values, dtype=config["dtype"])
    quality = jnp.nan_to_num(quality, nan=0.0)
    quality = jnp.where(quality == 0, 1e-5, quality)

    dependency_range = math.ceil(3 * D_m / upscale_resolution)
    mean_dist = 1 / jnp.mean(quality)
    alpha = upscale_resolution / mean_dist

    sensitivity_analyzer = SensitivityAnalysis(
        quality_raster=quality,
        permeability_raster=quality,
        distance=distance_fn,
        proximity=lambda dist: proximity(dist, D_m, alpha),
        coarsening_factor=0.,
        dependency_range=dependency_range,
        batch_size=config["batch_size"]
    )

    output = sensitivity_analyzer.run(sens_type)

    output_raster = deepcopy(quality_raster)
    output_raster.rio.set_crs(CRS_CH, inplace=True)
    output_raster.values = output
    switzerland_boundary = get_CH_border()

    output_raster = downscale(output_raster, suitability_dataset["mean_suitability"])
    output_raster = crop_raster(output_raster, switzerland_boundary)
    elasticity_raster = output_raster * suitability_dataset["mean_suitability"]
    output_path = Path(__file__).parent / Path(f"../../data/processed/{config['hash']}/elasticities") / hab / group
    file_name = f"elasticity_{sens_type}_{group}_{hab}.tif"
    output_path.mkdir(parents=True, exist_ok=True)
    elasticity_raster.rio.to_raster(output_path / file_name, compress='zstd')
    print("Saved elasticity raster at:", output_path / file_name)

if __name__ == "__main__":
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)

    config = {
        "batch_size": 16,
        "dtype": "float32",
        "analysis_precision": 2e-1, # percentage of the dispersal range
        "resolution": 25,            # meters
        "hash": sha
    }

    try:
        for sens_type in ["permeability", "quality"]:
            run_elasticity_analysis_for_group(args.group, args.hab, sens_type, config)
    except Exception as e:
        print(f"Failed to compute elasticity w.r.t. {sens_type} for {args.group}, {args.hab}: {str(e)}")

    print("Job completed.")
