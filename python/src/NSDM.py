# utilities for NSDM
from pathlib import Path
from utils_raster import load_raster
NSDM25m_PATH = Path(__file__, "../../../../data/NSDM_25m/").resolve()


class NSDM():
    def load_raster(self, species_name):
        path = NSDM25m_PATH/species_name
        raster_file = list(path.glob("*.tif"))
        if len(raster_file) == 1:
            raster_file = raster_file[0]
            raster = load_raster(raster_file)
            return raster

        else:
            raise ValueError(f"Problem reading {species_name} raster")

        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from TraitsCH import TraitsCH

    # Test initialization
    data = NSDM()
    species_name = "Squalius cephalus"
    raster = data.load_raster(species_name)
    raster.coarsen(x=10, y=10, boundary="trim").mean().plot()