# utilities for NSDM
from pathlib import Path
from utils_raster import load_raster
import socket
hostname = socket.gethostname()

if hostname == "gpunode05":
    NSDM_PATH = {25: Path("/shares/lud11/boussang/S2Z/data/NSDM_25m/"),
                100: Path("/shares/lud11/boussang/S2Z/data/NSDM_100m/")}
elif hostname == "MacBook-Pro-3.wsl.ch":
    NSDM_PATH = {25: Path(__file__, "../../../../data/NSDM_25m/").resolve(),
                100:Path(__file__, "../../../../data/NSDM_100m/").resolve()}

class NSDM:
    def load_raster(self, species_name, resolution=100):
        formatted_species_name = species_name.replace(" ", ".")
        
        filename_pattern = f"{resolution}m_{formatted_species_name}_reg_covariate_ensemble.tif"
        
        raster_file = list(NSDM_PATH[resolution].glob(filename_pattern))
        
        if len(raster_file) == 1:
            raster_file = raster_file[0]
            raster = load_raster(raster_file)
            raster = raster.rename(species_name)
            return raster
        else:
            raise ValueError(f"Problem reading raster for species '{species_name}'. File not found or multiple matches.")

if __name__ == "__main__":
    # Test initialization
    data = NSDM()
    species_name = "Squalius cephalus"
    raster = data.load_raster(species_name)
    raster.coarsen(x=10, y=10, boundary="trim").mean().plot()