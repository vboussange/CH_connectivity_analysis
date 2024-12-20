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
from jaxscape.resistance_distance import ResistanceDistance
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

def calculate_connectivity(hab_qual, activities, distance, D):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=hab_qual,
                     nb_active=activities.size)

    window_center = jnp.array([[activities.shape[0]//2+1, activities.shape[1]//2+1]])
    
    q = grid.array_to_node_values(hab_qual)
    dist = distance(grid, targets = window_center)

    K = jnp.exp(-dist/D) # calculating proximity matrix
    
    qKqT = hab_qual[window_center[0, 0], window_center[0, 1]] * (K.T @ q)
    connectivity = jnp.sum(qKqT)
    # TODO: for some reason, jnp.sqrt throws nans when differentiated
    
    # connectivity = jnp.sum(hab_qual[window_center[0, 0], window_center[0, 1]] * dist.T @ q)
    return connectivity

# TODO: check whether the gradient is correct - it seems that `hab_qual[window_center[0, 0], window_center[0, 1]]` is not working properly

connectivity_grad = eqx.filter_jit(eqx.filter_grad(calculate_connectivity))

run_calculation_vmap = eqx.filter_vmap(connectivity_grad, in_axes=(0,0,None,None))

@eqx.filter_jit
def batch_run_calculation(window_op, xy, hab_qual, activities, distance, D, raster_buffer):
    res = run_calculation_vmap(hab_qual, activities, distance, D)
    def scan_fn(raster_buffer, x):
        _xy, _rast = x
        raster_buffer = window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
        return raster_buffer, None
    raster_buffer, _ = lax.scan(scan_fn, raster_buffer, (xy, res))
    return raster_buffer

if __name__ == "__main__":
    
    # TODO: you may want to have an adapted buffer_size
    config = {"species_name": "Larix decidua",
              "batch_size": 1, # pixels, actual batch size is batch_size**2
              "resolution": 100, # meters
              "buffer_size_m": 10_000, # meters
              "dtype": "float32"
             }

    output_path = Path("output") / config["species_name"]
    output_path.mkdir(parents=True, exist_ok=True)
    traits = TraitsCH()
    
    # TODO: problem, Larix decidua has D_m = 95m
    # TODO: you should define low and high bounds for D_m
    D_m = traits.get_D(config["species_name"]) * 1000 # in meters

    quality_raster = compile_quality(config["species_name"], 
                                        D_m, 
                                        config["resolution"])
    
    
    quality = jnp.array(quality_raster.values[0,...], dtype=config["dtype"])
    quality = jnp.nan_to_num(quality, nan=0.0)
    
    # test
    # TODO: to remove
    # TODO: you may need to have a function that checks whether the raster is valid before running full computation
    quality = quality[1000:2000, 1000:2000]
    plt.imshow(quality)
    
    D = np.array(D_m / config["resolution"], dtype=config["dtype"])
    buffer_size = int(config["buffer_size_m"] / config["resolution"])
    distance = ResistanceDistance()

    # TODO: need to pad
    # quality_padded = padding(quality, buffer_size, config["batch_size"])
    
    batch_op = WindowOperation(
        shape=quality.shape, 
        window_size=config["batch_size"], 
        buffer_size=buffer_size
    )
    
    output = jnp.zeros_like(quality) # initialize raster
    window_op = WindowOperation(shape=(batch_op.total_window_size, batch_op.total_window_size), 
                                window_size=1, 
                                buffer_size=buffer_size)
    for (xy_batch, permeability_batch) in tqdm(batch_op.lazy_iterator(quality), desc="Batch progress", total=batch_op.nb_steps):
        if not jnp.all(jnp.isnan(permeability_batch)):
            xy, hab_qual = window_op.eager_iterator(permeability_batch)
            activities = jnp.ones_like(hab_qual, dtype="bool")
            raster_buffer = jnp.zeros_like(permeability_batch)
            res = batch_run_calculation(window_op, xy, hab_qual, activities, distance, D, raster_buffer)
            output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
    
    # TODO: need to unpad
    fig, ax = plt.subplots()
    cbar = ax.imshow(output)
    fig.colorbar(cbar, ax=ax)
    plt.show()
    
    # TODO: save output
    output_raster = deepcopy(quality_raster)
    output_raster.values[0,...] = output
    output_raster.rio.to_raster(output_path / "output.tif", compress='lzw')
    