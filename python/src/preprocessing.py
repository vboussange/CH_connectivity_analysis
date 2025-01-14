"""
Generating quality and permeability maps for a terrestrial species in CH
"""
# import pyproj
# required due to multiple pyproj installations
# pyproj.datadir.set_data_dir("/Users/victorboussange/projects/connectivity/connectivity_analysis/code/python/.env/share/proj/")
import xarray as xr
from shapely.geometry import box
from pathlib import Path
import numpy as np
from masks import MasksDataset, get_CH_border
from utils_raster import crop_raster, calculate_resolution, coarsen_raster, mask_raster, CRS_CH, fill_na_with_nearest
from TraitsCH import TraitsCH
from NSDM import NSDM
from EUSDM import EUSDM
from tqdm import tqdm

def compile_species_suitability(species_name, D_m, resolution):
    # loading fine resolution raster
    raster_NSDM = NSDM().load_raster(species_name)
    lat_resolution, lon_resolution = calculate_resolution(raster_NSDM)
    print(f"Original quality resolution: {lon_resolution/1000:0.3f}km")
    assert lat_resolution == lon_resolution
    resampling_factor = int(np.ceil(resolution/lat_resolution))
    raster_NSDM = coarsen_raster(raster_NSDM, resampling_factor)
    # raster_NSDM = raster_NSDM.fillna(0)
    
    # loading coarse resolution raster
    try:
        raster_coarse = EUSDM().load_raster(species_name)
        switzerland_boundary = get_CH_border()
        switzerland_buffer = switzerland_boundary.buffer(D_m)
        raster_coarse = crop_raster(raster_coarse.squeeze(), switzerland_buffer)
        
        # merging coarse and fine rasters 
        raster_coarse_interp = raster_coarse.interp_like(raster_NSDM, method="nearest")
        combined_raster = xr.where(raster_NSDM.notnull(), raster_NSDM, raster_coarse_interp)
        combined_raster.rio.set_crs(raster_NSDM.rio.crs, inplace=True)
        combined_raster = combined_raster.rename(species_name)
        
    except Exception as e:
        print(f"Failed to load coarse resolution raster: {e}")
        combined_raster = raster_NSDM
    
    # masking out land for aquatic species
    combined_raster = mask_raster(combined_raster, TraitsCH(), MasksDataset())
    return combined_raster


def compile_group_suitability(group, resolution):
    cache_path = Path(__file__).parent / Path(f"../.cache/{group}/suitability.nc")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        concatenated = xr.open_dataset(cache_path)
        res_lat, res_lon = calculate_resolution(concatenated)
        if res_lat == res_lon == resolution:
            return concatenated
    
    traits = TraitsCH()
    species = traits.get_all_species_from_group(group)
    if len(species) == 0:
        raise ValueError(f"No data found for group {group}")
    print(f"Group {group} has {len(species)} species")
    
    D_m = np.mean([traits.get_D(sp) for sp in species]) * 1000 # convert to meters
    switzerland_boundary = get_CH_border()
    switzerland_buffer = switzerland_boundary.buffer(D_m)
    minx, miny, maxx, maxy = switzerland_buffer.total_bounds

    # Loading fine and coarse resolution rasters
    nsdm_dataset = NSDM()
    nsdm_rasters = []
    for sp in tqdm(species, 
                   miniters=max(1, len(species)//100),
                    desc="Raster loading progress",
                   ):
        try:
            raster_fine = nsdm_dataset.load_raster(sp, resolution=resolution).rio.reproject(CRS_CH)
            res_lat, res_lon = calculate_resolution(raster_fine)
            assert res_lat == res_lon == resolution, f"Resolution mismatch for {sp}: {res_lat}x{res_lon}"
            raster_fine = raster_fine.rio.pad_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            nsdm_rasters.append(raster_fine)
        except Exception as e:
            print(e)
            continue
            # print(f"Failed to load raster for {sp}")
    print(f"Loaded {len(nsdm_rasters)} rasters")
    if len(nsdm_rasters) > 0:
        # Aggregating rasters
        nsdm_stack = xr.concat(nsdm_rasters, dim="species", join="left")
        mean_ch_sdm_suitability = nsdm_stack.mean(dim="species").squeeze("band").rename(species.iloc[0])    
        std_ch_sdm_suitability = nsdm_stack.std(dim="species").squeeze("band").rename(species.iloc[0])    

        # Reprojecting to desired resolution and masking out land for aquatic species
        rast = []
        for combined_raster, name in zip([mean_ch_sdm_suitability, std_ch_sdm_suitability], ["mean_suitability", "std_suitability"]):
            # interpolating data
            combined_raster = fill_na_with_nearest(combined_raster)
            combined_raster = mask_raster(combined_raster, traits, MasksDataset())
            combined_raster = combined_raster.rename(name)
            rast.append(combined_raster)
        
        # Merging to single dataset and saving
        concatenated = xr.merge(rast).astype("float32")
        concatenated = concatenated.rio.clip(switzerland_buffer, all_touched=True, drop=True)
        concatenated.rio.set_crs(CRS_CH, inplace=True)
        concatenated.attrs["D_m"] = D_m
        concatenated.attrs["N_species"] = len(nsdm_rasters)
        concatenated.attrs["species"] = [r.name for r in nsdm_rasters]

        concatenated.to_netcdf(cache_path)
        # save_to_netcdf(concatenated, cache_path, scale_factor=1000)
        # with open(cache_path, "wb") as f:
        #     pickle.dump((mean_suitability, std_suitability, D_m), f)
        
        return concatenated
    else:
        raise ValueError("No suitability raster found for group")

# TODO: fix epsilon
def compile_resistance(quality, barriers, eps =  1e-5):
    
    # initial resistance is uniquely based on quality
    initial_resistance = - np.log(quality + eps) 
    
    # adding extra resistances due to barriers
    final_resistance = initial_resistance.copy()
    for barrier in barriers:
        extra_resistances = xr.full_like(quality, fill_value=-np.log(eps))
        extra_resistances.rio.set_crs(quality.rio.crs)
        barrier = barrier.get_mask()
        extra_resistances = extra_resistances.rio.clip(barrier, all_touched=True, drop=False)
        
        final_resistance = xr.where(extra_resistances.notnull(), extra_resistances, final_resistance)
        # we filter out out of domain values
        final_resistance = xr.where(initial_resistance.notnull(), final_resistance, initial_resistance)
    
    return final_resistance.rio.set_crs(quality.rio.crs)