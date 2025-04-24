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
from utils_raster import calculate_resolution, CRS_CH, fill_na_with_nearest, load_geotiff_dataset, dataset_to_geotiff
from TraitsCH import TraitsCH
from NSDM import NSDM, NSDM_PATH
from tqdm import tqdm

from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance
import pandas as pd

# 17 groups
GROUP_INFO = {
            "Amphibians": EuclideanDistance(),
            "Bees": LCPDistance(),
            "Beetles": LCPDistance(),
            "Birds": EuclideanDistance(),
            "Bryophytes": EuclideanDistance(),
            "Mammals": LCPDistance(),
            "Reptiles": LCPDistance(),
            "Fishes": LCPDistance(),
            "Vascular_plants": EuclideanDistance(),
            "Spiders": LCPDistance(),
            "Dragonflies": LCPDistance(),
            "Grasshoppers": LCPDistance(),
            "Butterflies": LCPDistance(),
            "Fungi": EuclideanDistance(),
            "Molluscs": LCPDistance(),
            "Lichens": EuclideanDistance(),
            "May_stone_caddisflies": LCPDistance(),
            }

def compile_group_suitability(group, hab, resolution):
    """
    Incrementally compute mean and std of the suitability rasters for all species in a taxonomic group.
    """
    cache_path = Path(__file__).parent / Path(f"../../data/processed/{hab}/suitability_{resolution}m_{group}_{hab}.tif")
    if cache_path.exists():
        concatenated = load_geotiff_dataset(cache_path)
        res_lat, res_lon = calculate_resolution(concatenated)
        if res_lat == res_lon == resolution:
            return concatenated

    traits = TraitsCH()
    species = traits.get_all_species_from_group(group).to_list()
    
    species_in_hab = []
    for sp in species:
        habitats = traits.get_habitat(sp)
        # if both Aqu and Ter --> Aqu by default
        if len(habitats) > 1 and "Aqu" in habitats:
            habitats = ["Aqu"]
        if hab in habitats:
            species_in_hab.append(sp)
        
    if len(species) == 0:
        raise ValueError(f"No data found for group {group}")
    print(f"Group {group} has {len(species)} species")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate buffer distance, get Swiss boundary, and buffer
    nsdm_dataset = NSDM()


    # List all species with rasters available in NSDM_PATH for the given resolution
    print("Checking available species...")
    available_species = []
    species_data = []

    for sp in species_in_hab:
        formatted_species_name = sp.replace(" ", ".")
        filename_pattern = f"*{formatted_species_name}_reg_covariate_ensemble.tif"
        raster_file = list(NSDM_PATH[resolution].glob(filename_pattern))
        if len(raster_file) == 1:
            available_species.append(sp)
            dispersal_range = traits.get_D(sp)  # Get dispersal range for the species
            species_data.append({"species": sp, "dispersal_range_km": dispersal_range})

    # Save available species to a CSV file
    if len(available_species) == 0:
        raise ValueError(f"No rasters found for group {group}")
    print(f"Group has {len(available_species)} species with rasters available")
    
    available_species_path = Path(__file__).parent / Path(f"../../data/processed/{hab}/available_species_{group}_{hab}.csv")
    available_species_path.parent.mkdir(parents=True, exist_ok=True)
    species_df = pd.DataFrame(species_data)
    species_df.to_csv(available_species_path, index=False)

    D_m = species_df.dispersal_range_km.mean() * 1000  # convert to meters
    switzerland_boundary = get_CH_border()
    # padding to avoid edge effects
    switzerland_buffer = switzerland_boundary.buffer(3 * D_m)
    minx, miny, maxx, maxy = switzerland_buffer.total_bounds

    # Initialize aggregators
    sum_raster = None
    sumsq_raster = None
    valid_count = None
    loaded_species = []

    ref_raster = None  # This will be our alignment reference
    for sp in tqdm(
        available_species,
        miniters=max(1, len(available_species)//100),
        desc="Raster loading progress",
    ):
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
        
    # Compute mean and standard deviation
    # Avoid dividing by zero in places where valid_count = 0
    mean_raster = (sum_raster / valid_count.where(valid_count != 0)).where(valid_count != 0)
    var_raster = (sumsq_raster / valid_count.where(valid_count != 0)) - (mean_raster ** 2)
    std_raster = np.sqrt(var_raster.clip(min=0))  # clip negative rounding errors

    # Final fill (nearest), mask, rename, and squeeze the band dimension
    mask = MasksDataset()[hab]
    mean_raster_filled = fill_na_with_nearest(mean_raster)
    mean_raster_filled = mean_raster_filled.rio.clip(mask.geometry, all_touched=True, drop=True)
    mean_raster = mean_raster.where(~np.isnan(mean_raster), mean_raster_filled).rename("mean_suitability")
    
    std_raster_filled = fill_na_with_nearest(std_raster)
    std_raster_filled = std_raster_filled.rio.clip(mask.geometry, all_touched=True, drop=True)
    std_raster = std_raster.where(~np.isnan(std_raster), std_raster_filled).rename("std_suitability")

    # Merge into a single dataset
    concatenated = xr.merge([mean_raster, std_raster]).astype("float32")

    # Clip to Switzerland + buffer
    concatenated = concatenated.rio.clip(switzerland_buffer, all_touched=True, drop=True)
    concatenated.rio.set_crs(CRS_CH, inplace=True)

    # Store metadata
    concatenated.attrs["D_m"] = D_m
    concatenated.attrs["habitat"] = hab
    concatenated.attrs["group"] = group
    concatenated.attrs["species"] = loaded_species
    concatenated.attrs["N_species"] = len(loaded_species)

    # Save to cache
    dataset_to_geotiff(concatenated, cache_path)

    return concatenated

if __name__ == "__main__":
    # Example usage
    # group = "Mammals"
    # hab = "Aqu"
    # resolution = 25  # meters
    # suitability_dataset = compile_group_suitability(group, hab, resolution)
    
    # precalculating mean suitability maps for all groups
    import concurrent.futures
    resolution = 25  # meters
    tasks = [(group, hab, resolution) for group in GROUP_INFO for hab in ["Aqu", "Ter"]]

    def process_task(args):
        group, hab, resolution = args
        try:
            return compile_group_suitability(group, hab, resolution)
        except Exception as e:
            print(f"Error processing {group}, {hab}: {e}")
            return None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="Group-Habitat Processing"))
