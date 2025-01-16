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

from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance

GROUP_INFO = {"Mammals": LCPDistance(),
              "Reptiles": LCPDistance(),
              "Amphibians": EuclideanDistance(),
              "Birds": EuclideanDistance(),
              "Fishes": LCPDistance(),
              "Vascular_plants": EuclideanDistance(),
              "Bryophytes": EuclideanDistance(),
              "Spiders": LCPDistance(),
              "Insects": LCPDistance(),
              "Fungi": EuclideanDistance(),
              "Molluscs": LCPDistance(),
              "Lichens": EuclideanDistance()}

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
    """
    Incrementally compute mean and std of the suitability rasters for all species in a taxonomic group.
    """
    cache_path = Path(__file__).parent / Path(f"../../data/{group}/suitability_{resolution}m.nc")
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
    
    # Calculate buffer distance, get Swiss boundary, and buffer
    D_m = np.mean([traits.get_D(sp) for sp in species]) * 1000  # convert to meters
    switzerland_boundary = get_CH_border()
    switzerland_buffer = switzerland_boundary.buffer(D_m)
    minx, miny, maxx, maxy = switzerland_buffer.total_bounds

    nsdm_dataset = NSDM()

    # Initialize aggregators
    sum_raster = None
    sumsq_raster = None
    valid_count = None
    loaded_species = []

    ref_raster = None  # This will be our alignment reference

    for sp in tqdm(
        species,
        miniters=max(1, len(species)//100),
        desc="Raster loading progress",
    ):
        try:
            # Load the raster at the desired resolution
            raster_fine = nsdm_dataset.load_raster(sp, resolution=resolution)
            # Ensure it has the correct CRS
            raster_fine = raster_fine.rio.reproject(CRS_CH)
            
            # Check resolution
            res_lat, res_lon = calculate_resolution(raster_fine)
            assert res_lat == res_lon == resolution, f"Resolution mismatch for {sp}: {res_lat}x{res_lon}"
            
            # Pad so that each species raster has the same bounding box
            raster_fine = raster_fine.rio.pad_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

            # The first successfully loaded raster becomes our reference
            if ref_raster is None:
                ref_raster = raster_fine
            else:
                # Reproject subsequent rasters to match the reference
                raster_fine = raster_fine.rio.reproject_match(ref_raster)
            
            # Convert to float32 to save memory
            raster_fine = raster_fine.astype("float32")

            # If this is the first time, initialize sum_raster, sumsq_raster, valid_count
            if sum_raster is None:
                sum_raster = raster_fine.fillna(0.0)
                sumsq_raster = (raster_fine ** 2).fillna(0.0)
                valid_count = xr.where(raster_fine.notnull(), 1, 0)
            else:
                # Accumulate partial sums
                sum_raster = sum_raster + raster_fine.fillna(0.0)
                sumsq_raster = sumsq_raster + (raster_fine.fillna(0.0) ** 2)
                valid_count = valid_count + xr.where(raster_fine.notnull(), 1, 0)

            loaded_species.append(sp)

        except Exception as e:
            print(f"Failed to load raster for {sp}: {e}")
            continue

    # If we did not manage to load any rasters, abort
    if not loaded_species:
        raise ValueError("No suitability raster found for group")

    print(f"Loaded {len(loaded_species)} rasters successfully.")

    # Compute mean and standard deviation
    # Avoid dividing by zero in places where valid_count = 0
    mean_raster = (sum_raster / valid_count.where(valid_count != 0)).where(valid_count != 0)
    var_raster = (sumsq_raster / valid_count.where(valid_count != 0)) - (mean_raster ** 2)
    std_raster = np.sqrt(var_raster.clip(min=0))  # clip negative rounding errors

    # Final fill (nearest), mask, rename, and squeeze the band dimension
    mean_raster = fill_na_with_nearest(mean_raster)
    mean_raster = mask_raster(mean_raster.rename(loaded_species[0]), traits, MasksDataset()).rename("mean_suitability")

    std_raster = fill_na_with_nearest(std_raster)
    std_raster = mask_raster(std_raster.rename(loaded_species[0]), traits, MasksDataset()).rename("std_suitability")

    # Merge into a single dataset
    concatenated = xr.merge([mean_raster, std_raster]).astype("float32")

    # Clip to Switzerland + buffer
    concatenated = concatenated.rio.clip(switzerland_buffer, all_touched=True, drop=True)
    concatenated.rio.set_crs(CRS_CH, inplace=True)

    # Store metadata
    concatenated.attrs["D_m"] = D_m
    concatenated.attrs["N_species"] = len(loaded_species)
    concatenated.attrs["species"] = loaded_species

    # Save to NetCDF
    concatenated.to_netcdf(cache_path)

    return concatenated

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