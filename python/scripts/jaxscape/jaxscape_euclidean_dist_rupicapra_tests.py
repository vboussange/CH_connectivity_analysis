"""
Running sensitivity analysis of equivalent connected habitat for euclidean distance.
This script copies the behavior of omniscape.
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
import equinox
from tqdm import tqdm
import sys
import json
sys.path.append("./../../src")
from utils_raster import load_raster


def import_raster_and_info(species_name):
    path = Path("output") / species_name
    
    resistance = load_raster(path / "resistance.tif", scale=False)
    resistance = jnp.array(resistance[0,...], dtype="bfloat16")
    
    quality = load_raster(path / "quality.tif", scale=False)
    quality = jnp.array(quality[0, ...], dtype="bfloat16")

    
    info_path = path / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    return resistance, quality, info

def calculate_connectivity(habitat_permability, activities, distance, D):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=habitat_permability.size)

    window_center = jnp.array([[habitat_permability.shape[0]//2+1, habitat_permability.shape[1]//2+1]])
    
    q = grid.array_to_node_values(habitat_permability)
    dist = distance(grid, targets = window_center)
    K = jnp.exp(-dist/D) # calculating proximity matrix
    
    qKqT = habitat_permability[window_center[0, 0], window_center[0, 1]] * (K.T @ q)
    connectivity = jnp.sqrt(jnp.sum(qKqT))
    return connectivity

connectivity_grad = equinox.filter_jit(equinox.filter_grad(calculate_connectivity))

run_calculation_vmap = equinox.filter_vmap(connectivity_grad)

def batch_run_calculation(window_op, distance, raster, hab_qual, x_start, y_start, activities, D):
    return None
    # res = run_calculation_vmap(hab_qual, activities, distance, D)
    
    # TODO: replace by a lax.scan loop
    # but if you do that, you need to define xx and yy as static arguments
    # for i, (xx, yy) in enumerate(zip(x_start, y_start)):
    #     raster = window_op.add_window_to_raster(xx, yy, raster, res[i, ...])
    # return raster

if __name__ == "__main__":
    window_size = 1 # must be odd to be placed at the center
    batch_size = 10000
    
    result_path = Path("./results/WindowOperation2")
    result_path.mkdir(parents=True, exist_ok=True)
    species_name = "Rupicapra rupicapra"
    
    # resistance is not used for euclidean distance
    _, quality, sp_info = import_raster_and_info(species_name)
    
    D = int(sp_info["D_m"] / sp_info["resolution"]) + 1
    distance = EuclideanDistance()
    buffer_size = D

    window_op = WindowOperation(
        shape=quality.shape, 
        window_size=window_size, 
        buffer_size=buffer_size
    )

    raster = jnp.zeros_like(quality)

    # TODO: the iterator is very, very slow! We should improve this, benchmarking in a proper setting  
    # check this implementation: https://github.com/jax-ml/jax/issues/3171
    for (x_start, y_start, hab_qual) in tqdm(window_op.iterate_window_batches(quality, batch_size), desc="Batch progress"):
        activities = jnp.ones((hab_qual.shape[0], 2*buffer_size+window_size, 2*buffer_size+window_size), dtype="bool")
        raster = batch_run_calculation(window_op, 
                                       distance, 
                                       raster, 
                                       hab_qual, 
                                       x_start, 
                                       y_start, 
                                       activities, 
                                       D)

    fig, ax = plt.subplots()
    cbar = ax.imshow(raster)
    fig.colorbar(cbar, ax=ax)
    plt.show()
    # fig.savefig(result_path / "lcp_moving_window.png", dpi=400)
    
    # On RTX 3090:

    # N = 1000
    # window_size = 5 # must be odd to be placed at the center
    # buffer_size = 20
    # batch_size = 1000 
    # takes about 02:54min to run

    # N = 1120
    # window_size = 9 # must be odd
    # buffer_size = 20
    # batch_size = 500 
    # Takes about 01:28min to run