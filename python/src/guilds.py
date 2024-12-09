# this file reads data from spinfo.csv, but it is a bit of a doublon and you should rather use TraitsCH data

import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import box
from pathlib import Path
import netCDF4
import pandas as pd

GUILDS_EU_PATH = Path(__file__).parent / '../../../data/GUILDS_EU_SP/'

class GuildsData():
    def __init__(self):
        spinfo = pd.read_csv(GUILDS_EU_PATH / "GUILDS_EU_spinfo.csv", delimiter=";")
        spinfo.fillna(0, inplace=True)
        spinfo[spinfo.columns[6:-1]] = spinfo.iloc[:, 6:-1].astype(int)
        self.spinfo = spinfo
        # df = pd.read_csv(TRAITS_CH_PATH, delimiter=" ", na_values=["NA"])
        # # TODO: dummy place holder, to be changed
        # df["habitat"] = "Aqu"
        # self.data = df
        
    def get_key(self, species_name, key):
        df = self.spinfo
        species_row = df[df['species'] == species_name]
        if not species_row.empty:
            return species_row[key].values[0]
        else:
            raise ValueError(f"Species '{species_name}' not found in the dataset.")
        
    def get_guild(self, species):
        # return all guilds the species belong to, stored as dymmy variables in spinfo
        guilds = []
        for column in self.spinfo.columns[6:-1]:
            if column.startswith('GUILDE') and self.spinfo.loc[self.spinfo['species'] == species, column].values[0] == 1:
                guilds.append(column.replace('GUILDE.', ''))
        return guilds
    
    def get_species(self, guild_name):
        return self.spinfo[self.spinfo[guild_name] == 1].species
        
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
    import matplotlib.pyplot as plt
    from TraitsCH import TraitsCH

    # Test initialization
    guilds_data = GuildsData()
    species = "Squalius cephalus"
    raster = guilds_data.get_suitability(species)
    raster.coarsen(x=10, y=10, boundary="trim").mean().plot() # pass
    
    # get_species
    sp_list = guilds_data.get_species("GUILDE.11")
    guild = guilds_data.get_guild(species)
    
    # plot number of species per guild
    guild_counts = guilds_data.spinfo.iloc[:, 6:-1].sum()
    plt.figure(figsize=(10, 6))
    guild_counts.plot(kind='bar')
    plt.xlabel('Guild')
    plt.ylabel('Number of Species')
    plt.show()
    
    # for each guild, plot the distribution of dispersal distance.
    # you can get the dispersal distance like so:
    traits = TraitsCH()

    dispersal_distances = []
    for guild in guilds_data.spinfo.columns[6:-1]:
        species_in_guild = guilds_data.get_species(guild)
        distances = []
        for species in species_in_guild:
            try:
                distance = traits.get_D(species)
                distances.append(distance)
            except ValueError:
                continue
        dispersal_distances.append((guild, distances))

    plt.figure(figsize=(12, 8))
    for guild, distances in dispersal_distances:
        if distances:
            plt.hist(distances, bins=20, alpha=0.5, label=guild)
    plt.xlabel('Dispersal Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Dispersal Distances by Guild')
    plt.show()
    
    # can you get maximum dispersal distance
    max_dispersal_distances = {}
    for guild, distances in dispersal_distances:
        if distances:
            max_dispersal_distances[guild] = max(distances)

    print("Maximum dispersal distances by guild:")
    for guild, max_distance in max_dispersal_distances.items():
        print(f"{guild}: {max_distance}")