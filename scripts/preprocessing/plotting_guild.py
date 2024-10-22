import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd

dataset_path = Path(__file__).parent / '../../data/GUILDES_EU_buffer_dist=50km_resampling_4.nc'
dataset = xr.open_dataset(dataset_path, engine='netcdf4')

SWITZERLAND_BOUNDARY_PATH = Path(__file__).parent / '../../data/swiss_boundaries/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp'
switzerland_boundary = gpd.read_file(SWITZERLAND_BOUNDARY_PATH)


for guild in list(dataset.data_vars):
    fig, ax = plt.subplots()
    dataset[guild].plot(ax=ax, add_colorbar=False, add_labels=False)
    switzerland_boundary.boundary.plot(ax=ax, color='tab:red')  # Use boundary to plot only the outline
    ax.axis('off')
    fig.savefig(dataset_path.parent / f"GUILDES_EU/plots/{guild}.png", dpi=300, transparent=True)
