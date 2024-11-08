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
from jaxscape.gridgraph import GridGraph, QUEEN_CONTIGUITY, ExplicitGridGraph
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.rsp_distance import RSPDistance, rsp_distance
from jaxscape.utils import well_adapted_movement
from jaxscape.rastergraph import RasterGraph

import jax
import jax.numpy as jnp
import sys
sys.path.append("./../src")
# from swissTLMRegio import MasksDataset, get_canton_border
from TraitsCH import TraitsCH
from utils_raster import NSDM25m_PATH, load_raster, CRS_CH, calculate_resolution, coarsen_raster
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

PATH_CHAMOIS_LOC = '../../../data/chamois_genetics/chamois_sampling_locations.csv'
PATH_CHAMOIS_GEN_DIST = '../../../data/chamois_genetics/genetic_distances_chamois.RDS'

def load_chamois_data():
    df = pd.read_csv(PATH_CHAMOIS_LOC)
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    gdf = gdf.to_crs(CRS_CH)
    gdf['Latitude'] = gdf.geometry.y
    gdf['Longitude'] = gdf.geometry.x

    
    gend = pyreadr.read_r(PATH_CHAMOIS_GEN_DIST)[None] # also works for RData
    
    return gdf, gend


# jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    buffer_distance = 0 # meters
    resolution = 10_000 # meters
    output_file = Path("../../../data/processed/NSDM25m/") / f"rescaled_NSDM25m_{resolution}m.nc"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    traits_dataset = TraitsCH()
    gdf, gend = load_chamois_data()

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
    
    raster = cropped_and_coarsened_rasters[0]
    habitat_quality = jnp.array(raster.data[0,:,:])
    plt.imshow(habitat_quality)
    activities = ~jnp.isnan(habitat_quality) & (habitat_quality > 0.)
    plt.imshow(activities)
    gridgraph = GridGraph(activities = activities, 
                          vertex_weights = habitat_quality)
    A = gridgraph.get_adjacency_matrix(QUEEN_CONTIGUITY)
    # distance matrix calculation
    distance = EuclideanDistance(res=lat_resolution)
    eucliddist_mat = distance(gridgraph)
    # plotting 
    vertex_index = gridgraph.coord_to_active_vertex_index(habitat_quality.shape[0]//2, habitat_quality.shape[1]//2)
    dist_to_vertex = gridgraph.node_values_to_array(eucliddist_mat[:, vertex_index]) # broken test
    plt.imshow(dist_to_vertex)

    eucliderastergraph = RasterGraph(ExplicitGridGraph(activities, 
                                           habitat_quality,
                                           adjacency_matrix=eucliddist_mat), raster.x.data, raster.y.data)
    
    
    # distance matrix calculation
    theta = jnp.array(1.)
    distance = RSPDistance(theta=theta)
    RSPdist_mat = distance(gridgraph)
    # plotting 
    vertex_index = gridgraph.coord_to_active_vertex_index(10, 18)
    dist_to_vertex = gridgraph.node_values_to_array(RSPdist_mat[:, vertex_index])
    plt.imshow(dist_to_vertex)
    
    RSPrastergraph = RasterGraph(ExplicitGridGraph(activities, 
                                           habitat_quality,
                                           adjacency_matrix=RSPdist_mat), raster.x.data, raster.y.data)
    
    loc = (jnp.array(gdf.Longitude), jnp.array(gdf.Latitude))
    
    i, j = RSPrastergraph.index(*loc)
    active_individuals=gridgraph.activities[i, j]
    loc_active_individuals = (loc[0][active_individuals], loc[1][active_individuals])
    
    
    RSP_dist_mat_ind = RSPrastergraph.get_distance(loc_active_individuals)
    euclid_dist_mat_ind = eucliderastergraph.get_distance(loc_active_individuals)
    gen_mat_ind = gend.loc[gdf.Sample_ID[np.array(active_individuals)], gdf.Sample_ID[np.array(active_individuals)]].to_numpy()
    
    mask = ~np.eye(gen_mat_ind.shape[0], dtype=bool)

    plt.scatter(RSP_dist_mat_ind[mask], gen_mat_ind[mask])
    plt.scatter(euclid_dist_mat_ind[mask], gen_mat_ind[mask])
    plt.scatter(euclid_dist_mat_ind[mask], RSP_dist_mat_ind[mask])

    # dataset = xr.open_dataset(output_file, engine='netcdf4', decode_coords="all")
    # TODO: calculate correlation
    # TODO: make an optimization to get best matching theta value