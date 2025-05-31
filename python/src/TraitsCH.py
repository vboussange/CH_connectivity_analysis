import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd

TRAITS_CH_PATH = Path(__file__).parent / '../../data/raw/TraitsCH_30052025'

DICT_STUDY_GROUP = {"mammals": "Mammals", "reptiles": "Reptiles", "amphibians": "Amphibians", "birds": "Birds", "fishes": "Fishes",
               "vascular_plants": "Vascular_plants", "bryophytes": "Bryophytes", "spiders": "Spiders", "coleoptera": "Beetles",
               "odonata": "Dragonflies", "orthoptera": "Grasshoppers", "lepidoptera": "Butterflies", "bees": "Bees", "fungi": "Fungi",
               "molluscs": "Molluscs", "lichens": "Lichens", "may_stone_caddis_flies": "May_stone_caddisflies"}


class TraitsCH():
    def __init__(self):
        file = TRAITS_CH_PATH / "1_SDMapCHv1_CH3Div_TerAqu.csv"
        data = pd.read_csv(file, index_col=0)
        # Replace '/' with '_' in all string columns
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype(str).str.replace('/', '_', regex=False)
        self.data = data
        
    def get_row(self, species_name):
        df = self.data
        species_row = df[df['species'] == species_name]
        if not species_row.empty:
            return species_row
        else:
            raise ValueError(f"Species '{species_name}' not found in the dataset.")
        
    def get_key(self, species_name, key):
        species_row = self.get_row(species_name)
        if key in species_row.columns:
            return species_row[key].values[0]
        else:
            raise ValueError(f"Key not found in the dataset.")
        
    def get_D(self, species_name):
        # TODO: check for realistic values
        return self.get_key(species_name, "Dispersal_km")
        
    def get_all_species_from_group(self, group_name):
        return self.data[self.data["CH3Div.group"] == group_name].species
    
    def get_habitat(self, species_name):
        return self.get_row(species_name)["CH3Div.scheme"].values[0]
        

if __name__ == "__main__":
    # Test initialization
    traits = TraitsCH()
    species = "Squalius cephalus"
    dispersal_distance = traits.get_D(species)
    group = traits.get_habitat(species)
    species_list = traits.get_all_species_from_group("amphibians")
