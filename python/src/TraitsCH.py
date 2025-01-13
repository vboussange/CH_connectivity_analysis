import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd

TRAITS_CH_PATH = Path(__file__).parent / '../../../data/TraitsCH/dispersal_focus/s2z_compiled'
GUILDS_EU_PATH = Path(__file__).parent / '../../../data/GUILDS_EU_SP/'

class TraitsCH():
    def __init__(self):
        all_files = TRAITS_CH_PATH.glob("*.txt")
        df_list = []
        for file in all_files:
            df = pd.read_csv(file, delimiter=" ", na_values=["NA"])
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df[["Fos", "Ter", "Aqu", "Arb", "Aer"]] = df[["Fos", "Ter", "Aqu", "Arb", "Aer"]].astype(bool)
        df[df.columns[14:40]] = df[df.columns[14:40]].astype(bool)

        self.data = df
        
    def get_row(self, species_name):
        df = self.data
        species_row = df[df['Species'] == species_name]
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
    
    def _get_dummy(self, species, columns):
        # return all guilds the species belong to, stored as dummy variables in spinfo
        data = []
        species_row = self.get_row(species)
        for col in columns:
            if species_row[col].values[0] == 1:
                data.append(col)
        return data
    
    def get_all_species_from_guild(self, guild_name):
        return self.data[self.data[guild_name] == 1].Species
    
    def get_all_species_from_group(self, group_name):
        return self.data[self.data["Group"] == group_name].Species
    
    def get_habitat(self, species_name):
        return self._get_dummy(species_name, ["Fos", "Ter", "Aqu", "Arb", "Aer"])[0]
    
    def get_guilds(self, species_name):
        return self._get_dummy(species_name, self.data.columns[14:40])

    def get_suitability(self, species_name):
        path = GUILDS_EU_PATH/species_name
        raster_file = list(path.glob("*.tif"))
        if len(raster_file) == 1:
            raster_file = raster_file[0]
            raster = rioxarray.open_rasterio(raster_file, mask_and_scale=True)
            raster = raster.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
            raster = raster.rename(path.parent.stem)
            return raster
        if (raster.max() > 1) & (raster.min() < 100):
            print("Rescaling habitat quality between 0 and 1")
            raster = raster / 100.

        else:
            ValueError("Problem reading {species_name} raster")
        

if __name__ == "__main__":
    # Test initialization
    traits = TraitsCH()
    species = "Squalius cephalus"
    habitat = traits.get_habitats(species)  # Replace with actual species name
    assert habitat[0] == "Aqu"
    dispersal_distance = traits.get_D(species)
    guilds = traits.get_guilds(species)


    species = "Lagopus muta"
    habitat = traits.get_habitats(species)  # Replace with actual species name
    assert habitat[0] == "Ter"
    dispersal_distance = traits.get_D(species)
    guilds = traits.get_guilds(species)
