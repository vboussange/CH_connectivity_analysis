"""
Computing genetic vs ecological vs euclidean distance.

"""

import pyreadr
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd
import numpy as np
from jaxscape.gridgraph import GridGraph
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.rsp_distance import RSPDistance
from jaxscape.rastergraph import RasterGraph
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
sys.path.append("./../src")
# from swissTLMRegio import MasksDataset, get_canton_border
from TraitsCH import TraitsCH
from utils_raster import NSDM25m_PATH, load_raster, CRS_CH, calculate_resolution, coarsen_raster


if __name__ == "__main__":
    buffer_distance = 0 # meters
    resolution = 10_000 # meters
    output_file = Path("../../../data/processed/NSDM25m/") / f"rescaled_NSDM25m_{resolution}m.nc"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    traits_dataset = TraitsCH()
    gend = pyreadr.read_r('../../../data/chamois_genetics/genetic_distances_chamois.RDS') # also works for RData

    # raster_files = list(Path(input_dir).glob('**/*.tif'))
    raster_files = [Path(NSDM25m_PATH, "Rupicapra.rupicapra_reg_covariate_ensemble.tif")]
    rasters = [load_raster(file) for file in raster_files]
    # reprojections
    rasters = [rast.rio.reproject(CRS_CH) for rast in rasters]
    

    print("Original raster of resolution:")
    lat_resolution, lon_resolution = calculate_resolution(rasters[0])
    print(f"Latitude resolution: {lat_resolution/1000:0.3f}km")
    print(f"Longitude resolution: {lon_resolution/1000:0.3f}km")
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    cropped_and_coarsened_rasters = [coarsen_raster(raster, resampling_factor) for raster in rasters]
    print("Coarse raster of resolution:")
    lat_resolution, lon_resolution = calculate_resolution(cropped_and_coarsened_rasters[0])
    print(f"Latitude resolution: {lat_resolution/1000:0.2f}km")
    print(f"Longitude resolution: {lon_resolution/1000:0.2f}km")
    
    # print(f"saved at {output_file}")
    # dataset = xr.merge(cropped_and_coarsened_raster, join="left")
    # dataset.to_netcdf(output_file, engine='netcdf4')
    
    habitat_quality = jnp.array(cropped_and_coarsened_rasters[0].data[0,:,:])
    plt.imshow(habitat_quality)
    activities = ~jnp.isnan(habitat_quality)
    plt.imshow(activities)
    gridgraph = GridGraph(activities = activities, vertex_weights = habitat_quality)
    A = gridgraph.get_adjacency_matrix()
    # distance matrix calculation
    distance = EuclideanDistance(res=lat_resolution)
    eucliddist_mat = distance(gridgraph)
    
    # plotting 
    vertex_index = gridgraph.coord_to_active_vertex_index(10, 18)
    dist_to_vertex = gridgraph.node_values_to_array(eucliddist_mat[:, vertex_index]) # broken test
    plt.imshow(dist_to_vertex)

    
    
    # distance matrix calculation
    theta = jnp.array(2.)
    distance = RSPDistance(theta=lat_resolution)
    
    # TODO: this is not working, need to investigate e.g. the cost matrix
    RSPdist_mat = distance(gridgraph)
    
    # plotting 
    vertex_index = gridgraph.coord_to_active_vertex_index(10, 18)
    dist_to_vertex = gridgraph.node_values_to_array(RSPdist_mat[:, vertex_index]) # broken test
    plt.imshow(dist_to_vertex)
    # dataset = xr.open_dataset(output_file, engine='netcdf4', decode_coords="all")