# NOTE: This script is best implemented in jaxscape/benchmark
import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np  # for NaN handling, not used in heavy computations
import matplotlib.pyplot as plt
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
from jaxscape.utils import BCOO_to_sparse, get_largest_component_label
import jax
from jax.experimental.sparse import BCOO
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix      
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from jaxscape.moving_window import WindowOperation
import xarray as xr
from pathlib import Path
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.rsp_distance import RSPDistance

def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution       

def get_valid_activities(hab_qual, activities):
    # TODO: the best would be to avoid transfer between numpy and jax array
    grid = GridGraph(activities, hab_qual)
    A = grid.get_adjacency_matrix()
    Anp = BCOO_to_sparse(A)
    _, labels = connected_components(Anp, directed=True, connection="strong")
    label = get_largest_component_label(labels)
    vertex_belongs_to_largest_component_node = labels == label
    activities_pruned = grid.node_values_to_raster(vertex_belongs_to_largest_component_node)
    activities_pruned = activities_pruned == True
    return activities_pruned

def run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance):
    """Performs the sensitivity analysis on each valid window.
    `D` must be expressed in the unit of habitat quality in `window_op`.
    """
    sensitivity_raster = jnp.full_like(habitat_quality_raster, jnp.nan)
    for x_start, y_start, hab_qual in tqdm(window_op.iterate_windows(habitat_quality_raster), total=window_op.nb_steps, desc="Running Analysis"):
        # Build grid graph and calculate Euclidean distances
        activities = hab_qual > 0
        valid_activities = get_valid_activities(hab_qual, activities)

        # TODO: we should jit the whole block below instead of jitting at each iteration
        def calculate_ech(habitat_quality):
            grid = GridGraph(activities=activities, vertex_weights=habitat_quality)
            dist = distance(grid)
            # scaling
            dist = dist / dist.max()
            proximity = jnp.exp(-dist / D)
            landscape = ExplicitGridGraph(activities=activities, 
                                        vertex_weights=habitat_quality, 
                                        adjacency_matrix=proximity)
            ech = landscape.equivalent_connected_habitat()
            return ech
        grad_ech = jax.jit(jax.grad(calculate_ech))
        sensitivities = grad_ech(habitat_suitability)

        window_op.update_raster_from_window(x_start, y_start, sensitivity_raster, sensitivities)

    return sensitivity_raster

def load_habitat_suitability(sp_name, path_ncfile = Path("data/large_extent_habitat_suitability.nc")):
    with xr.open_dataset(path_ncfile, engine="netcdf4", decode_coords="all") as da: 
        habitat_suitability = da[sp_name] / 100
        da.close()
    res = calculate_resolution(da)
    jax_raster = jnp.array(habitat_suitability.data[0,:,:])
    jax_raster = jnp.nan_to_num(jax_raster, nan=0.)
    return jax_raster, res

# Example usage
if __name__ == "__main__":

    sp_name = "Salmo trutta"
    D_km = 1.0 #traits.get_D(sp_name)
    
    params_computation = {"window_size": 1.}
    alpha = jnp.array(21.)
    habitat_suitability, res = load_habitat_suitability(sp_name)
    D = D_km / alpha

    window_op = WindowOperation(shape = habitat_suitability.shape, 
                                window_size = params_computation["window_size"], 
                                buffer_size = int(3 * D_km / res))
    
    distance = EuclideanDistance(res=1.)
    sensitivity_raster = run_sensitivity_analysis(habitat_suitability, window_op, D, distance, params_computation["cut_off"])
    
    theta = jnp.array(0.01)
    distance = RSPDistance(theta)
    output_array = run_analysis(window_op, D, distance)
    # output_array = run_analysis(window_op, D, RSPDistance.rsp_distance, theta=theta)
