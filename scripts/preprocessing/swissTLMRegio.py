import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd
import pyogrio


SWISSTLMREGIO_PATH = Path(__file__).parent / '../../../data/swisstlmregio_2024_2056_gpkg/swissTLMRegio_Product_LV95.gpkg'
layers = pyogrio.list_layers(SWISSTLMREGIO_PATH)

class AquaticMaskDataset:
    def __init__(self, buffer_distance=500):
        self.path = SWISSTLMREGIO_PATH
        output_file = SWISSTLMREGIO_PATH.parent / f"Aqu_mask_buffer_distance={buffer_distance}.gpkg"
        if output_file.is_file():
            self.aquatic_mask = gpd.read_file(output_file, driver="GPKG")
        else:
            print("Creating cached aquatic mask...")
            self.create_aquatic_mask(buffer_distance)
            print(f"Aquatic mask cached at {output_file.resolve()}.")
    
    def create_aquatic_mask(self, buffer_distance):
        layers = pyogrio.list_layers(self.path)
        
        # Load river and lake layers
        gdf_rivers = gpd.read_file(self.path, layer="tlmregio_hydrography_flowingwater")
        gdf_lakes = gpd.read_file(self.path, layer="tlmregio_hydrography_lake")
        
        # Apply buffer to the rivers
        gdf_rivers = gdf_rivers.buffer(buffer_distance)
        
        # Concatenate river and lake geometries
        gdf_aqu = pd.concat([gdf_rivers.geometry, gdf_lakes.geometry])
        
        # Merge the geometries into a single mask
        gdf_aqu_merged = gpd.GeoSeries(gdf_aqu.union_all(), crs=gdf_rivers.crs)
        
        # Save the result to a GeoPackage file
        output_file = SWISSTLMREGIO_PATH.parent / f"Aqu_mask_buffer_distance={buffer_distance}.gpkg"
        gdf_aqu_merged.to_file(output_file, driver="GPKG")
        
        # Store the mask in the object
        self.aquatic_mask = gdf_aqu_merged


class MaskDataset():
    # an instantiation mask_dataset should return an AquaticMaskDataset with mask_dataset["Aqu"]
