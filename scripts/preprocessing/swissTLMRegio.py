import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd
import pyogrio


SWISSTLMREGIO_PATH = Path(__file__).parent / '../../../data/swisstlmregio_2024_2056_gpkg/swissTLMRegio_Product_LV95.gpkg'
SWISSTLMREGIO_BOUNDARIES_PATH = Path(__file__).parent / '../../../data/swisstlmregio_2024_2056_gpkg/swissTLMRegio_BOUNDARIES_LV95.gpkg'

# layers = pyogrio.list_layers(SWISSTLMREGIO_PATH)

def get_CH_border():
    # layers = pyogrio.list_layers(SWISSTLMREGIO_BOUNDARIES_PATH)
    gdf = gpd.read_file(SWISSTLMREGIO_BOUNDARIES_PATH, layer="swisstlmregio_landesgebiet")
    return gdf[gdf.icc == "CH"].geometry

    

class BaseMaskDataset:
    def __init__(self, dataset_path, output_file_name, buffer_distance=0):
        self.path = dataset_path
        self.output_file = dataset_path.parent / output_file_name
        self.buffer_distance = buffer_distance

        if self.output_file.is_file():
            self.mask = gpd.read_file(self.output_file, driver="GPKG")
        else:
            print(f"Creating cached mask: {output_file_name}...")
            self.create_mask()
            print(f"Mask cached at {self.output_file.resolve()}.")
    
    def create_mask(self):
        raise NotImplementedError("Subclasses must implement the 'create_mask' method.")
    
    def get_mask(self):
        """Retrieve the mask, assuming it's been loaded or created."""
        return self.mask.geometry


class AquaticMaskDataset(BaseMaskDataset):
    def __init__(self, buffer_distance=500):
        super().__init__(SWISSTLMREGIO_PATH, f"Aqu_mask_buffer_distance={buffer_distance}.gpkg", buffer_distance)

    def create_mask(self):        
        # Load river and lake layers
        gdf_rivers = gpd.read_file(self.path, layer="tlmregio_hydrography_flowingwater")
        gdf_lakes = gpd.read_file(self.path, layer="tlmregio_hydrography_lake")
        
        # Apply buffer to the rivers
        gdf_rivers = gdf_rivers.buffer(self.buffer_distance)
        
        # Concatenate river and lake geometries
        gdf_aqu = pd.concat([gdf_rivers.geometry, gdf_lakes.geometry])
        
        # Merge the geometries into a single mask
        gdf_aqu_merged = gpd.GeoSeries(gdf_aqu.union_all(), crs=gdf_rivers.crs)
        gdf_aqu_merged.to_file(self.output_file, driver="GPKG")
        
        # Store the mask in the object
        self.mask = gdf_aqu_merged
        
class MasksDataset:
    def __init__(self, buffer_distance=500):
        # Create a dictionary of mask datasets. More masks can be added in the future.
        self.masks = {
            "Aqu": AquaticMaskDataset(buffer_distance=buffer_distance),
            # "Road": RoadMaskDataset(buffer_distance=buffer_distance)  # Example of another mask
        }

    def __getitem__(self, key):
        # Provide access to different mask datasets using string keys
        if key in self.masks:
            return self.masks[key].get_mask()
        else:
            raise KeyError(f"Mask '{key}' not found.")


# class RoadMaskDataset(BaseMaskDataset):
#     def __init__(self, buffer_distance=100):
#         super().__init__(SWISSTLMREGIO_PATH, f"Road_mask_buffer_distance={buffer_distance}.gpkg", buffer_distance)

#     def create_mask(self):
#         # Load road layers
#         gdf_roads = gpd.read_file(self.path, layer="tlmregio_transport_roads")
        
#         # Apply buffer to the roads
#         gdf_roads = gdf_roads.buffer(self.buffer_distance)
        
#         # Merge the road geometries into a single mask
#         gdf_road_merged = gpd.GeoSeries(gdf_roads.union_all(), crs=gdf_roads.crs)
        
#         # Save the result to a GeoPackage file
#         gdf_road_merged.to_file(self.output_file, driver="GPKG")
        
#         # Store the mask in the object
#         self.mask = gdf_road_merged
