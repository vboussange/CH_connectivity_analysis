# Calculate ecological connectivity index for Ter and Aqu species, and combined.


import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append(str(Path(__file__).parent / Path("../src/")))
from group_preprocessing import GROUP_INFO
from NSDM import NSDM_PATH
from utils_raster import load_raster, CRS_CH
import pandas as pd
REF_RASTER = load_raster(NSDM_PATH[25] / "Rattus.norvegicus_reg_covariate_ensemble.tif")

def rescale(data):
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)

def safe_add(a, b):
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, a + b))

def plot_raster(rast, title, path):
            fig, ax = plt.subplots()
            plot_data = rast.coarsen(x=10, y=10, boundary="trim").mean()
            plot_data.plot(
                ax=ax,
                # vmax=0.5,
                cmap="magma",
                cbar_kwargs={"label": title,
                            "shrink": 0.3},
            )
            ax.set_aspect("equal")
            ax.set_title("")
            ax.set_axis_off()
            # fig.tight_layout()
            fig.savefig(path, dpi=300)

def calculate_ecis(hab, base_path, aggregation):
    """
    Calculate ecological connectivity importance score for multiple taxonomic groups, 
    and also return a DataFrame containing D_m for each group.
    """

    total_elasticity = None

    for group in GROUP_INFO:
        logger.info("Processing %s species for group: %s", hab, group)
        path_elasticities = base_path / "elasticities" / hab / group
        tif_files = list(path_elasticities.glob("*.tif"))
        elasticities = {}

        for tif_file in tif_files:
            logger.info("Reading file: %s", tif_file)
            key = tif_file.stem
            elasticities[key] = rioxarray.open_rasterio(tif_file)

        if len(elasticities) > 0:
            group_summed_elasticity = elasticities[f"elasticity_quality_{group}_{hab}"]
            if len(elasticities) > 1:
                group_summed_elasticity += elasticities[f"elasticity_permeability_{group}_{hab}"]
                
            group_summed_elasticity = rescale(np.log(group_summed_elasticity+1e-5))

            if total_elasticity is None:
                total_elasticity = group_summed_elasticity
            elif aggregation == "max":
                total_elasticity = xr.ufuncs.fmax(total_elasticity, group_summed_elasticity)
            elif aggregation == "mean":
                total_elasticity = xr.apply_ufunc(safe_add, total_elasticity, group_summed_elasticity)

    logger.info("Scaling the summed elasticity values from 0 to 1")
    total_elasticity = rescale(total_elasticity)
    return total_elasticity

if __name__ == "__main__":
    config = {"hash": "277b08f",
              "aggregation": "max",}
    base_path = Path(__file__).parent / Path("../../data/processed")  / config["hash"]
    ecis_path = base_path / "ecological_connectivity_importance_score"
    ecis_path.mkdir(parents=True, exist_ok=True)
    for hab in ["Aqu", "Ter"]:
        print("Processing type: ", hab)
        ecis = calculate_ecis(hab, base_path, config["aggregation"]).rio.set_crs(CRS_CH)
        ecis = ecis.rio.reproject_match(REF_RASTER)
        # Save result
        out_file = ecis_path / f"ecological_connectivity_importance_score_{config['aggregation']}_{hab}"
        logger.info("Saving final raster to: %s", out_file)
        ecis.rio.to_raster(str(out_file) + ".tif", compress="zstd")
        
        if True:
            plot_raster(ecis, f"Ecological Connectivity Importance Score ({hab})", out_file.with_suffix(".png"))


    # Combine Aqu and Ter rasters based on the aggregation method
    logger.info("Combining Aqu and Ter rasters using aggregation: %s", config["aggregation"])
    aqu_raster = rioxarray.open_rasterio(str(ecis_path / f"ecological_connectivity_importance_score_{config['aggregation']}_Aqu.tif"))
    ter_raster = rioxarray.open_rasterio(str(ecis_path / f"ecological_connectivity_importance_score_{config['aggregation']}_Ter.tif"))

    if config["aggregation"] == "max":
        combined_raster = rescale(xr.ufuncs.fmax(ter_raster, aqu_raster))
    elif config["aggregation"] == "mean":
        combined_raster = rescale(xr.apply_ufunc(safe_add, aqu_raster, ter_raster))

    # Save the combined raster
    combined_raster = combined_raster.rio.set_crs(CRS_CH)
    combined_out_file = ecis_path / f"ecological_connectivity_importance_score_{config['aggregation']}_Aqu_Ter"
    logger.info("Saving combined raster to: %s", combined_out_file.with_suffix(".tif"))
    combined_raster.rio.to_raster(str(combined_out_file.with_suffix(".tif")), compress="zstd")
    plot_raster(ecis, f"Ecological Connectivity Importance Score", combined_out_file.with_suffix(".png"))