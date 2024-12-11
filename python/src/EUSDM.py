# utilities for EU SDM

# this file reads data from spinfo.csv, but it is a bit of a doublon and you should rather use TraitsCH data

from pathlib import Path
from utils_raster import load_raster, CRS_CH

GUILDS_EU_PATH = Path(__file__).parent / '../../../data/GUILDS_EU_SP/'

class EUSDM():
    def load_raster(self, species_name):
        path = GUILDS_EU_PATH/species_name
        raster_file = list(path.glob("*.tif"))
        if len(raster_file) == 1:
            raster_file = raster_file[0]
            raster = load_raster(raster_file)
            return raster.rio.reproject(CRS_CH)

        else:
            raise ValueError(f"Problem reading {species_name} raster")

        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from TraitsCH import TraitsCH

    # Test initialization
    data = EUSDM()
    species_name = "Sylvia borin"
    raster = data.load_raster(species_name)
    raster.coarsen(x=10, y=10, boundary="trim").mean().plot()