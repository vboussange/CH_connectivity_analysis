import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
# from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent / Path("../../src/")))
from processing import GROUP_INFO

# def load_and_scale_elasticity(file_path):
#     """Load elasticity raster and scale the values."""
#     raster = rioxarray.open_rasterio(file_path)
    # values = raster.values
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(values.reshape(-1, 1)).reshape(values.shape)
    # return scaled_values

base_path = Path("output")

summed_elasticity = None
GROUP_INFO = {"Reptiles": None}
for group in GROUP_INFO:
    elasticity_quality_path = base_path / group / "elasticity_quality.tif"
    elasticity_permeability_path = base_path / group / "elasticity_permeability.tif"
    
    # Load and scale elasticity rasters
    elasticity_quality = rioxarray.open_rasterio(elasticity_quality_path)
    elasticity_permeability = rioxarray.open_rasterio(elasticity_permeability_path)
    ech = elasticity_quality.sum()
    
    # Sum the scaled elasticities
    if summed_elasticity is None:
        summed_elasticity = (elasticity_quality + elasticity_permeability) / ech
    else:
        summed_elasticity += (elasticity_quality + elasticity_permeability) / ech

# Save the summed elasticity raster
summed_elasticity.rio.to_raster(base_path / "elasticities_all.tif", compress='lzw')

# if __name__ == "__main__":
#     main()