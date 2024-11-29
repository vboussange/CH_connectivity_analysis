"""
Running sensitivity analysis for Rupicapra rupicapra.
"""
import gc
import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
import matplotlib.pyplot as plt
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
from tqdm import tqdm
import xarray as xr
from pathlib import Path
from jaxscape.moving_window import WindowOperation
from jaxscape.lcp_distance import LCPDistance
from jaxscape.rsp_distance import RSPDistance
from jaxscape.resistance_distance import ResistanceDistance
import equinox as eqx
import sys
sys.path.append("./../../src")
from TraitsCH import TraitsCH
from utils_raster import NSDM25m_PATH, load_raster, CRS_CH, calculate_resolution, coarsen_raster
import jax

def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution

def calculate_ech(habitat_quality, activities, D, distance, nb_active):
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_quality, 
                     nb_active=nb_active)
    dist = distance(grid)
    proximity = jnp.exp(-dist / D)
    landscape = ExplicitGridGraph(activities=activities, 
                                  vertex_weights=habitat_quality, 
                                  adjacency_matrix=proximity,
                                  nb_active=nb_active)
    ech = landscape.equivalent_connected_habitat()
    return ech

def run_sensitivity_analysis(habitat_quality, window_op, D, distance, threshold=0.1):
    sensitivity_raster = jnp.full_like(habitat_quality, jnp.nan)
    grad_ech = eqx.filter_jit(eqx.filter_grad(calculate_ech))
    
    for x_start, y_start, hab_qual in tqdm(window_op.iterate_windows(habitat_quality), total=window_op.nb_steps, desc="Running Analysis"):
        activities = hab_qual > threshold
        nb_active = int(jnp.sum(activities))
        # NOTE: due to jit compilation, it may be beneficial to have nb_active =
        # window_size**2 to avoid recompilation of the function. This does not
        # affect calculation of distances in the case of LCPDistance, but it
        # could.
        if nb_active > 0:
            sensitivities = grad_ech(hab_qual, activities, D, distance, nb_active)
            sensitivity_raster = window_op.update_raster_from_window(x_start, y_start, sensitivity_raster, sensitivities)
            del sensitivities
            gc.collect()
            
    return sensitivity_raster

def load_habitat_quality(resolution=1000.):
    habitat_quality = load_raster(Path(NSDM25m_PATH, "Rupicapra.rupicapra_reg_covariate_ensemble.tif"))
    habitat_quality = habitat_quality.rio.reproject(CRS_CH)
    lat_resolution, lon_resolution = calculate_resolution(habitat_quality)
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    habitat_quality = coarsen_raster(habitat_quality, resampling_factor)
    return jnp.array(habitat_quality.values[0,:,:])

if __name__ == "__main__":
    # data
    params_computation = {"window_size": 5,
                        "threshold": 0.1,
                        "resolution": 1000.,
                        "result_path": Path("./results")}
    
    sp_name = "Rupicapra rupicapra"
    traits_dataset = TraitsCH()
    D_m = jnp.array(traits_dataset.get_D(sp_name)) * 1000.
    habitat_quality = load_habitat_quality(resolution=params_computation["resolution"])

    alpha = jnp.array(1.0)
    resolution = jnp.array(1000.)
    D = D_m / alpha
    
    window_op = WindowOperation(shape=habitat_quality.shape, 
                                window_size=params_computation["window_size"], 
                                buffer_size=int(3 * D / resolution))
    
    plt.imshow(habitat_quality)
    print(f"We have {jnp.sum(habitat_quality > params_computation["threshold"])} active cells.")
    print(f"We have {window_op.total_window_size**2} active cells in window.")

    
    distance = ResistanceDistance()
    sensitivity_raster = run_sensitivity_analysis(habitat_quality, 
                                                  window_op, 
                                                  D, 
                                                  distance, 
                                                  threshold=params_computation["threshold"])
    plt.imshow(sensitivity_raster)

    # Save the sensitivity_raster as a numpy array
    sensitivity_raster_np = jax.device_get(sensitivity_raster)
    jnp.save("sensitivity_raster.npy", sensitivity_raster_np)
    
    # theta = jnp.array(0.01)
    # distance = RSPDistance(theta)
    # output_array = run_sensitivity_analysis(habitat_quality, window_op, D, distance)
    # plt.imshow(output_array)
