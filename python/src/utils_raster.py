import xarray as xr
import rioxarray
from pathlib import Path


NSDM25m_PATH = Path(__file__, "../../../../data/NSDM_25m/").resolve()
CRS_CH = "EPSG:2056" # https://epsg.io/2056

def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution

def coarsen_raster(raster, resampling_factor):
    raster_coarse = raster.coarsen(x=resampling_factor, y=resampling_factor, boundary='trim').mean()
    raster_coarse.rio.set_crs(raster.rio.crs)
    return raster_coarse
    
def load_raster(path):
    # Load the raster file
    raster = rioxarray.open_rasterio(path, mask_and_scale=True)
    raster = raster.drop_vars(["band"]) # we keep `spatial_ref` var. as it contains crs data
    raster = raster.rename(path.parent.stem)
    if (raster.max() > 1) & (raster.min() < 100):
        print("Rescaling habitat quality between 0 and 1")
        raster = raster / 100.
    return raster

def crop_raster(raster, buffer):
    buffered_gdf = gpd.GeoDataFrame(geometry=buffer)
    masked_raster = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    return masked_raster


def mask_raster(raster, traits_dataset, masks_dataset):
    sp_name = raster.name
    hab = traits_dataset.get_habitat(sp_name)
    if hab == "Aqu":
        mask = masks_dataset[hab]
        raster_masked = raster.rio.clip(mask, all_touched=True, drop=True)
        
    else:
        raster_masked = raster
    return raster_masked
