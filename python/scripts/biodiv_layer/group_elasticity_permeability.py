"""
Calculating the elasticity of habitat quality with respect to permeability using Jaxscape.
TODO: need to verify that the batching and calculation are correct.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use the first GPU

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
sys.path.append(str(Path(__file__).parent / Path("../../src/")))
from preprocessing import compile_group_suitability
from processing import batch_run_calculation, padding, GROUP_INFO
from postprocessing import postprocess
import xarray as xr
import rioxarray
from copy import deepcopy

def qKqT(permeability, hab_qual, activities, distance, D):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability,
                     nb_active=activities.size)

    window_center = jnp.array([[activities.shape[0]//2+1, activities.shape[1]//2+1]])
    q = grid.array_to_node_values(hab_qual)
    dist = distance(grid, sources=window_center).reshape(-1)

    K = jnp.exp(-dist/D) # calculating proximity matrix
    
    qKqT = hab_qual[window_center[0, 0], window_center[0, 1]] * (K.T @ q)

    return qKqT


qKqT_grad = eqx.filter_jit(eqx.filter_grad(qKqT))

qKqT_grad_vmap = eqx.filter_vmap(qKqT_grad, in_axes=(0, 0, 0, None, None))


if __name__ == "__main__":
    
    config = {"batch_size": 1, # pixels, actual batch size is batch_size**2
            "resolution": 100, # meters
            # percentage of the dispersal range, used to calculate landmarks
            # if the dispersal range is 10 pixels and the coarsening factor is 0.3, then the landmarks will be calculated every 2 pixels
            # each pixel should be involved by at least one landmark
            "coarsening_factor": 0.3,
            "dtype": "float32",
            }

    # # TODO: test to remove
    # GROUP_INFO = {"Reptiles": LCPDistance()}
    for group in GROUP_INFO:
        print("Computing elasticity for group:", group)
        distance = GROUP_INFO[group]

        output_path = Path("output") / group
        output_path.mkdir(parents=True, exist_ok=True)
        
        suitability_dataset = compile_group_suitability(group, 
                                                        config["resolution"])
        D_m = suitability_dataset.attrs["D_m"]
        quality = jnp.array(suitability_dataset["mean_suitability"].values, dtype=config["dtype"])
        quality = jnp.nan_to_num(quality, nan=0.0)
        quality = jnp.where(quality == 0, 1e-5, quality)
        assert jnp.all(quality > 0) and jnp.all(quality < 1) and jnp.all(jnp.isfinite(quality)), "Quality values must be between 0 and 1."
        
        ## Calculating meta parameters
        # disepersal in pixels
        D = np.array(D_m / config["resolution"], dtype=config["dtype"])
        assert D >= 1, "Dispersal range must be greater than 1 pixel."
        
        # number of pixels -1 to skip per iteration and that will not be considered as landmarks
        coarsening = int(jnp.ceil(D * config["coarsening_factor"]))
        if coarsening % 2 == 0:
            coarsening += 1
            
        # buffer size should be of the order of the dispersal range - half that of the window operation size
        # size distance is calculated from the center pixel of the window
        buffer_size = int(D - (coarsening - 1)/2)
        if buffer_size < 1:
            raise ValueError("Buffer size is too small. Consider decreasing the coarsening factor or decreasing the raster resolution.")
        
        batch_window_size = config["batch_size"] * coarsening

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
                permeability = hab_qual
                res = batch_run_calculation(batch_op, window_op, xy, qKqT_grad_vmap, permeability, hab_qual, activities, distance, D)
                output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
        
        # unpadding
        output = output[:quality.shape[0], :quality.shape[1]]
        
        elasticity = output * quality # quality == permeability

        # TODO: to remove
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # im1 = axes[0].imshow(quality)
        # axes[0].set_title("Quality")
        # fig.colorbar(im1, ax=axes[0], shrink=0.1)
        # im2 = axes[1].imshow(elasticity)
        # axes[1].set_title("Elasticity w.r.t permeability")
        # fig.colorbar(im2, ax=axes[1], shrink=0.1)
        # plt.show()
        
        # TODO: tif and nc files not consistent
        output_raster = deepcopy(suitability_dataset["mean_suitability"])
        output_raster.values = elasticity
        output_raster = postprocess(output_raster)
        output_raster.rio.to_raster(output_path / "elasticity_permeability.tif", compress='lzw')
        print("Saved elasticity raster at:", output_path / "elasticity_permeability.tif")
