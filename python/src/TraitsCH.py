import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd
import swissTLMRegio

TRAITS_CH_PATH = Path(__file__).parent / '../../../data/TraitsCH/dispersal_focus/S_MEAN_dispersal_all.txt'

class TraitsCH():
    def __init__(self):
        self.path = TRAITS_CH_PATH
        df = pd.read_csv(TRAITS_CH_PATH, delimiter=" ", na_values=["NA"])
        # TODO: dummy place holder, to be changed
        df["habitat"] = "Aqu"
        self.data = df
        
    def get_key(self, species_name, key):
        df = self.data
        species_row = df[df['Species'] == species_name]
        if not species_row.empty:
            return species_row[key].values[0]
        else:
            raise ValueError(f"Species '{species_name}' not found in the dataset.")
        
    def get_habitat(self, species_name):
        return self.get_key(species_name, "habitat")
        
    def get_D(self, species_name):
        # TODO: check for realistic values
        return self.get_key(species_name, "Dispersal_km")
        

if __name__ == "__main__":
    # Test initialization
    traits = TraitsCH()
    species = "Salmo trutta"
    habitat = traits.get_habitat(species)  # Replace with actual species name
    assert habitat == "Aqu"
    dispersal_distance = traits.get_D(species)
    assert dispersal_distance == 23.3632
    
    try:
        traits.get_habitat("non_existent_species")
    except ValueError as ve:
        print(f"Expected error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")