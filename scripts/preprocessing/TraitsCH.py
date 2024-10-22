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
        
    def get_habitat(self, species_name):
        df = self.data
        species_row = df[df['Species'] == species_name]
        if not species_row.empty:
            return species_row['habitat'].values[0]
        else:
            raise ValueError(f"Species '{species_name}' not found in the dataset.")