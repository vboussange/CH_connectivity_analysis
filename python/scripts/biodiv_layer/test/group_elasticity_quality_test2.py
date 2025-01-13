"""
Calculating the elasticity of habitat quality with respect to permeability using Jaxscape.
TODO: need to verify that the batching and calculation are correct.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU

import jax
import numpy as np
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.gridgraph import GridGraph, QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance

import equinox as eqx
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent / Path("../../../src/")))
from preprocessing import compile_group_suitability, CRS_CH
from processing import batch_run_calculation, padding, GROUP_INFO
from postprocessing import postprocess
import xarray as xr
import rioxarray
from copy import deepcopy
os.chdir(Path(__file__).parent)

def Kq(hab_qual, activities, distance, D):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=hab_qual,
                     nb_active=activities.size,
                     fun= lambda x, y: (x + y)/2,
                     neighbors=QUEEN_CONTIGUITY)

    window_center = jnp.array([[activities.shape[0]//2, activities.shape[1]//2]])
        
    # x_core_window, y_core_window = jnp.meshgrid(jnp.arange(window_op.buffer_size, 
    #                                                         window_op.window_size+window_op.buffer_size), 
    #                                             jnp.arange(window_op.buffer_size, 
    #                                                         window_op.window_size+window_op.buffer_size), indexing="xy")
    # core_window_indices = grid.coord_to_active_vertex_index(x_core_window.flatten(), y_core_window.flatten())

    dist = distance(grid, sources=window_center).reshape(-1)
    # dist = dist.at[core_window_indices].set(jnp.mean(dist[core_window_indices]))

    K = jnp.exp(-dist/D) # calculating proximity matrix
    
    
    core_window_qual = lax.dynamic_slice(hab_qual, 
                                start_indices=(window_op.buffer_size, window_op.buffer_size), 
                                slice_sizes=(window_op.window_size, window_op.window_size))


    # epsilon = K * jnp.sum(core_window_qual > 0) 
    epsilon = jnp.ones_like(K)
    epsilon = grid.node_values_to_array(epsilon)
    # epsilon = epsilon.at[window_center[0][0], window_center[0][1]].set(0)

    return epsilon

Kq_vmap = eqx.filter_vmap(Kq, in_axes=(0,0,None,None))

if __name__ == "__main__":
    
    config = {"batch_size": 2**4, # pixels, actual batch size is batch_size**2
            "resolution": 100, # meters
            # percentage of the dispersal range, used to calculate landmarks
            # if the dispersal range is 10 pixels and the coarsening factor is 0.3, then the landmarks will be calculated every 2 pixels
            # each pixel should be involved by at least one landmark
            "dtype": "float32",
            }
    

    distance = EuclideanDistance()
    quality = jnp.ones((100,100))     
    
    ## Calculating meta parameters
    # dispersal in pixels
    D = 20
    assert D >= 1, "Dispersal range must be greater than 1 pixel."
    print("Dispersal D:", D)

    # number of pixels -1 to skip per iteration and that will not be considered as landmarks
    coarsening = 3
    # coarsening = int(jnp.ceil(D * config["coarsening_factor"]))
    # if coarsening % 2 == 0:
    #     coarsening += 1
    # print("Coarsening pixels:", coarsening)
        
    # buffer size should be of the order of the dispersal range - half that of the window operation size
    # size distance is calculated from the center pixel of the window
    # we want buffer size to be a multiple of the coarsening factor
    buffer_size = int(D)
    buffer_size += coarsening - (buffer_size % coarsening)
    if buffer_size < 1:
        raise ValueError("Buffer size is too small. Consider decreasing the coarsening factor or decreasing the raster resolution.")
    
    batch_window_size = config["batch_size"]

    quality_padded = padding(quality, buffer_size, batch_window_size)
    
    batch_op = WindowOperation(
        shape=quality_padded.shape, 
        window_size=batch_window_size, 
        buffer_size=buffer_size)
    
    output = jnp.zeros_like(quality_padded) # initialize raster
    window_op = WindowOperation(shape=(batch_op.total_window_size, batch_op.total_window_size), 
                                window_size=coarsening, 
                                buffer_size=buffer_size)
    for (xy_batch, permeability_batch) in tqdm(batch_op.lazy_iterator(quality_padded), desc="Batch progress", total=batch_op.nb_steps):
        if not jnp.all(jnp.isnan(permeability_batch)):
            xy, hab_qual = window_op.eager_iterator(permeability_batch)
            activities = jnp.ones_like(hab_qual, dtype="bool")
            raster_buffer = jnp.zeros_like(permeability_batch)
            res = batch_run_calculation(batch_op, window_op, xy, Kq_vmap, hab_qual, activities, distance, D)
            output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
    
    # unpadding
    output = output[:quality.shape[0], :quality.shape[1]]
    
    # elasticity = output * quality
    plt.imshow(output[buffer_size:-buffer_size, buffer_size:-buffer_size])
    plt.colorbar()
    # plt.imshow(quality)