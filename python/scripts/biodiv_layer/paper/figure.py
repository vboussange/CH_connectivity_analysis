import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

base_path = Path("../output")
group = "Reptiles"
elasticity_quality_path = base_path / group / "elasticity_quality.tif"
elasticity_permeability_path = base_path / group / "elasticity_permeability.tif"

# Load and scale elasticity rasters
elasticity_quality = rioxarray.open_rasterio(elasticity_quality_path, mask_and_scale=True).squeeze()
elasticity_permeability = rioxarray.open_rasterio(elasticity_permeability_path, mask_and_scale=True).squeeze()
summed_elasticity = (elasticity_quality + elasticity_permeability)
scaler = MinMaxScaler()
summed_elasticity_scaled = scaler.fit_transform(summed_elasticity.values.reshape(-1, 1)).reshape(summed_elasticity.shape)
summed_elasticity = xr.DataArray(summed_elasticity_scaled, dims=summed_elasticity.dims, coords=summed_elasticity.coords)
summed_elasticity = summed_elasticity.coarsen(x=5, y=5, boundary="trim").mean().rolling(x=2, y=2).mean()
size_fig = 8
fig, ax = plt.subplots(figsize=(size_fig, size_fig * (summed_elasticity.shape[0] / summed_elasticity.shape[1])))
cbar = summed_elasticity.plot(ax=ax, 
                              cmap = "magma",
                              cbar_kwargs={
                                            # "label": "Pixel importance for connectivity",
                                           "shrink": 0.5})
cbar.colorbar.ax.yaxis.set_label_position('left')
cbar.colorbar.ax.yaxis.set_ticks_position('right')
# ax.set_title("Connectivity importance for reptiles")
ax.set_title("")
ax.axis('off')
fig.tight_layout()
fig.savefig("elasticity_reptiles.png", dpi=150)
# if __name__ == "__main__":
#     main()