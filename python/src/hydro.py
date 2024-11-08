import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd

RIVERS_CH_PATH = Path(__file__).parent / '../../../data/hydro/GIS/RS_Width.shp'

class HabitatDataset():
    def __init__(self, tol=0.01):
        self.path = RIVERS_CH_PATH
        path_merged_rivers = RIVERS_CH_PATH.parent / (str(RIVERS_CH_PATH.stem) + f"_merged_tol_{tol}.geojson")
        if path_merged_rivers.is_file():
            self.rivers = gpd.read_file(path_merged_rivers, driver='ESRI Shapefile')
        else:
            print("Creating cached river mask...")
            gpd_rivers = gpd.read_file(RIVERS_CH_PATH)
            gpd_rivers = gpd_rivers.to_crs(CRS)
            river_mask = gpd_rivers.union_all()
            merged_gdf_river = gpd.GeoDataFrame(geometry=[river_mask], crs=CRS)
            simplified_merged_gdf_river = merged_gdf_river.simplify(tolerance=tol, preserve_topology=True)
            simplified_merged_gdf_river.to_file(path_merged_rivers, driver='ESRI Shapefile')
            print(f"River mask cached at {path_merged_rivers.resolve()}.")

            self.rivers = merged_gdf_river