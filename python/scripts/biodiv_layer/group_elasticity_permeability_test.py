"""
Testing elasticity script.
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
from preprocessing import compile_group_suitability
from processing import batch_run_calculation, padding
import xarray as xr
import rioxarray
from copy import deepcopy

def qKqT(permeability, hab_qual, activities, distance, D):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability,
                     nb_active=activities.size)

    window_center = (activities.shape[0]//2+1, activities.shape[1]//2+1)

    return permeability[window_center[0], window_center[1]] 


qKqT_grad = eqx.filter_jit(eqx.filter_grad(qKqT))

qKqT_grad_vmap = eqx.filter_vmap(qKqT_grad, in_axes=(0, 0, 0, None, None))


if __name__ == "__main__":
    
    config = {"group": "Reptiles",
              "batch_size": 2**6, # pixels, actual batch size is batch_size**2
              "resolution": 100, # meters
              "coarsening_factor": 3, # pixels, must be odd, where 1 is no coarsening
              # TODO: coarsening_factor may require tuning w.r.t. the dispersal range
              "dtype": "float32",
             }

    distance = LCPDistance()

    output_path = Path("output") / config["group"]
    output_path.mkdir(parents=True, exist_ok=True)
    
    quality_raster, _, D_m = compile_group_suitability(config["group"], 
                                                    config["resolution"])
    
    quality = jnp.array(quality_raster.values, dtype=config["dtype"])
    quality = jnp.nan_to_num(quality, nan=0.0)
    
    quality = quality[1000:1500, 1000:1500]
    
    D = np.array(3 * D_m / config["resolution"], dtype=config["dtype"])

    # buffer size should be of the order of the dispersal range - half that of the window operation size
    # size distance is calculated from the center pixel of the window
    buffer_size = int(D - (config["coarsening_factor"] - 1)/2)
    if buffer_size < 1:
        raise ValueError("Buffer size is too small. Consider decreasing the coarsening factor or decreasing the raster resolution.")
    
    batch_window_size = config["batch_size"] * config["coarsening_factor"]

    quality_padded = padding(quality, buffer_size, batch_window_size)
    
    batch_op = WindowOperation(
        shape=quality_padded.shape, 
        window_size=batch_window_size, 
        buffer_size=buffer_size)
    
    output = jnp.zeros_like(quality_padded) # initialize raster
    window_op = WindowOperation(shape=(batch_op.total_window_size, batch_op.total_window_size), 
                                window_size=config["coarsening_factor"], 
                                buffer_size=buffer_size)
    for (xy_batch, permeability_batch) in tqdm(batch_op.lazy_iterator(quality_padded), desc="Batch progress", total=batch_op.nb_steps):
        if not jnp.all(jnp.isnan(permeability_batch)):
            xy, hab_qual = window_op.eager_iterator(permeability_batch)
            activities = jnp.ones_like(hab_qual, dtype="bool")
            permeability = hab_qual
            res = batch_run_calculation(batch_op, window_op, xy, qKqT_grad_vmap, permeability, hab_qual, activities, distance, D)
            output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
    
    # unpadding
    output = output[buffer_size:-buffer_size, buffer_size:-buffer_size]
    # test
    print("Testing output...")
    x_idx = jnp.arange(config["coarsening_factor"]-1, output.shape[0], config["coarsening_factor"])
    y_idx = jnp.arange(config["coarsening_factor"]-1, output.shape[1], config["coarsening_factor"])
    assert jnp.all(output[x_idx[:, None], y_idx] == 1)
    assert jnp.allclose(output[output < 1], 0)
    print("Test passed")