# Calculate the overall group-summed elasticity for multiple taxonomic groups,
# excluding fishes, based on per-group TIF files. The function reads and sums
# "elasticity_quality" and optionally "elasticity_permeability" for each group,
# averages them, and ultimately scales the final summed elasticity values from 0
# to 1.


import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append(str(Path(__file__).parent / Path("../src/")))
from group_preprocessing import GROUP_INFO

def calculate_summed_elasticities(config, base_path):
    """
    Calculate the overall group-summed elasticity for multiple taxonomic groups, 
    excluding fishes, based on per-group TIF files. The function reads and sums 
    "elasticity_quality" and optionally "elasticity_permeability" for each group, 
    averages them, and ultimately scales the final summed elasticity values from 0 to 1.
    """
    summed_elasticity = None

    for group in GROUP_INFO:
        if group == "Fishes":
            continue
        logger.info("Processing group: %s", group)
        path_elasticities = base_path / config["hash"] / group
        tif_files = list(path_elasticities.glob("*.tif"))
        elasticities = {}

        for tif_file in tif_files:
            logger.info("Reading file: %s", tif_file)
            key = tif_file.stem
            elasticities[key] = rioxarray.open_rasterio(tif_file)

        # Calculate group-summed elasticity
        ech = elasticities["elasticity_quality"].sum()
        group_summed_elasticity = elasticities["elasticity_quality"]
        if len(elasticities) > 1:
            group_summed_elasticity += elasticities["elasticity_permeability"]
        group_summed_elasticity = group_summed_elasticity / ech / len(elasticities)

        # Aggregate into the overall sum
        if summed_elasticity is None:
            summed_elasticity = group_summed_elasticity
        else:
            summed_elasticity += group_summed_elasticity

    # Scale from 0 to 1
    logger.info("Scaling the summed elasticity values from 0 to 1")
    summed_elasticity_min = summed_elasticity.min()
    summed_elasticity_max = summed_elasticity.max()
    summed_elasticity = (
        (summed_elasticity - summed_elasticity_min)
        / (summed_elasticity_max - summed_elasticity_min)
    )

    return summed_elasticity

if __name__ == "__main__":
    config = {"hash": "88a8982"}
    base_path = Path(__file__).parent / Path("../../data/processed")
    summed_elasticity = calculate_summed_elasticities(config, base_path)
    # Save result
    out_file = base_path / config["hash"] / "elasticities_all.tif"
    logger.info("Saving final raster to: %s", out_file)
    summed_elasticity.rio.to_raster(out_file, compress="lzw")
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_data = summed_elasticity.coarsen(x=10, y=10, boundary="trim").mean()
        plot_data.plot(
            ax=ax,
            vmax=0.5,
            cmap="magma",
            cbar_kwargs={"label": "Importance for ecological connectivity",
                         "shrink": 0.3},
        )
        ax.set_aspect("equal")
        ax.set_title("")
        ax.set_axis_off()
        # fig.tight_layout()
        fig.savefig(Path(__file__).parent / "../../ecological_connectivity_importance.png", dpi=300)
