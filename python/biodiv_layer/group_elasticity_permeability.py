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

def proximity(dist):
    return jnp.exp(-dist) / jnp.sum(jnp.exp(-dist))

def run_elasticity_analysis_for_group(group, aquatic, config):
    """
    Runs elasticity analysis for a single group using the given configuration.
    """
    distance_fn = GROUP_INFO[group]
    if isinstance(distance_fn, EuclideanDistance):
        return
    
    if aquatic:
        typ = "Aquatic"
    else:
        typ = "Terrestrial"

    output_path = Path(__file__).parent / Path(f"../../data/processed/{config['hash']}") / typ / group
    file_name = f"elasticity_permeability_{group}_{typ}.tif"
    output_path.mkdir(parents=True, exist_ok=True)

    suitability_dataset = compile_group_suitability(group, aquatic, config["resolution"])
    fine_resolution, _ = calculate_resolution(suitability_dataset["mean_suitability"])
    D_m = suitability_dataset.attrs["D_m"]

    upscale_resolution = max(fine_resolution, D_m * config["analysis_precision"])
    quality_raster = upscale(suitability_dataset["mean_suitability"], upscale_resolution)

    quality = jnp.array(quality_raster.values, dtype=config["dtype"])
    quality = jnp.nan_to_num(quality, nan=0.0)
    quality = jnp.where(quality == 0, 1e-5, quality)

    dependency_range = math.ceil(D_m / upscale_resolution)
    if dependency_range < 1:
        return

    sensitivity_analyzer = SensitivityAnalysis(
        quality_raster=quality,
        permeability_raster=quality,
        distance=distance_fn,
        proximity=proximity,
        coarsening_factor=0.,
        dependency_range=dependency_range,
        batch_size=config["batch_size"]
    )

    output = sensitivity_analyzer.run(d_permeability_vmap)

    output_raster = deepcopy(quality_raster)
    output_raster.rio.set_crs(CRS_CH, inplace=True)
    output_raster.values = output
    switzerland_boundary = get_CH_border()

    output_raster = downscale(output_raster, suitability_dataset["mean_suitability"])
    output_raster = crop_raster(output_raster, switzerland_boundary)
    elasticity_raster = output_raster * suitability_dataset["mean_suitability"]
    # TODO: for Aquatic, need to mask out lakes
    elasticity_raster.rio.to_raster(output_path / file_name, compress='zstd')
    print("Saved elasticity raster at:", output_path / file_name)


if __name__ == "__main__":
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head, short=True)

    config = {
        "batch_size": 32,
        "dtype": "float32",
        "analysis_precision": 1e-1, # percentage of the dispersal range
        "resolution": 25,            # meters
        "hash": sha
    }

    for group in GROUP_INFO:
        for aquatic in [True, False]:
            print("Computing permeability elasticity for group:", group)
            try:
                run_elasticity_analysis_for_group(group, aquatic, config)
            except Exception as e:
                print(f"Failed to compute elasticity for group {group} with aquatic={aquatic}: {e}")

    print("Finished job.")