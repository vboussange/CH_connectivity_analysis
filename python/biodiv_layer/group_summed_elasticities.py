import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_summed_elasticities(config, base_path):
    summed_elasticity = None
    GROUP_INFO = {"Reptiles": None}

    for group in GROUP_INFO:
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
    config = {"hash": "ace93a1"}
    base_path = Path("results")
    summed_elasticity = calculate_summed_elasticities(config, base_path)
    # Save result
    out_file = base_path / config["hash"] / "elasticities_all.tif"
    logger.info("Saving final raster to: %s", out_file)
    summed_elasticity.rio.to_raster(out_file, compress="lzw")
