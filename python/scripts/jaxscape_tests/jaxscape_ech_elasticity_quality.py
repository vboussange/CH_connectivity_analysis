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
from jaxscape.lcp_distance import LCPDistance

import equinox as eqx
from tqdm import tqdm
import sys
sys.path.append("./../../src")
from utils_raster import load_raster
from preprocessing import compile_quality, padding
from TraitsCH import TraitsCH
import xarray as xr
import rioxarray
from copy import deepcopy

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

@eqx.filter_jit
def batch_run_calculation(window_op, xy, hab_qual, activities, distance, D, raster_buffer):
    res = Kq_vmap(hab_qual, activities, distance, D)
    def scan_fn(raster_buffer, x):
        _xy, _rast = x
        raster_buffer = window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
        return raster_buffer, None
    raster_buffer, _ = lax.scan(scan_fn, raster_buffer, (xy, res))
    return raster_buffer

if __name__ == "__main__":
    
    # TODO: you may want to have a buffer_size adapted to dispersal range
    config = {"species_name": "Rupicapra rupicapra",
              "batch_size": 9, # pixels, actual batch size is batch_size**2
              "resolution": 100, # meters
            #   "buffer_size_m": 5_000, # meters
              "coarsening_factor": 9, # must be odd, where 1 is no coarsening
              "dtype": "float32",
             }

    distance = LCPDistance()

    output_path = Path("output") / config["species_name"]
    output_path.mkdir(parents=True, exist_ok=True)
    traits = TraitsCH()
    
    D_m = traits.get_D(config["species_name"]) * 1000 # in meters

    quality_raster = compile_quality(config["species_name"], 
                                        D_m, 
                                        config["resolution"])
    
    quality = jnp.array(quality_raster.values[0,...], dtype=config["dtype"])
    quality = jnp.nan_to_num(quality, nan=0.0)
    
    # test
    # TODO: to remove
    # TODO: you may need to have a function that checks whether the raster is valid before running full computation
    # quality = quality[1000:2000, 1000:2000]
    plt.imshow(quality)
    
    D = np.array(D_m / config["resolution"], dtype=config["dtype"])

    # buffer size should be of the order of the dispersal range - half that of the window operation size
    # size distance is calculated from the center pixel of the window
    buffer_size = int(D - (config["coarsening_factor"] - 1)/2)
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
            raster_buffer = jnp.zeros_like(permeability_batch)
            res = batch_run_calculation(window_op, xy, hab_qual, activities, distance, D, raster_buffer)
            output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
    
    # unpadding
    output = output[:quality.shape[0], :quality.shape[1]]
    
    # TODO: need to calculate qKq
    qKq = output * quality
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(quality)
    axes[0].set_title("Quality")
    fig.colorbar(im1, ax=axes[0], shrink=0.1)
    im2 = axes[1].imshow(qKq)
    axes[1].set_title("qKq")
    fig.colorbar(im2, ax=axes[1], shrink=0.1)
    plt.show()
    
    # TODO: save output
    output_raster = deepcopy(quality_raster)
    output_raster.values[0,...] = qKq
    output_raster.rio.to_raster(output_path / "output.tif", compress='lzw')
    