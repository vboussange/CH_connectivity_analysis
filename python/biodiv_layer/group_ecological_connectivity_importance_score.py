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
from group_preprocessing import GROUP_INFO, compile_group_suitability
import pandas as pd
def calculate_ecis(config, hab, base_path):
    """
    Calculate ecological connectivity importance score for multiple taxonomic groups, 
    and also return a DataFrame containing D_m for each group.
    """

    def rescale(data):
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min)

    dm_records = []
    total_elasticity = None

    for group in GROUP_INFO:
        logger.info("Processing %s species for group: %s", hab, group)
        path_elasticities = base_path / config["hash"] / hab / group
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
                
            group_summed_elasticity = np.log(rescale(group_summed_elasticity)+ 1e-5)

            if total_elasticity is None:
                total_elasticity = group_summed_elasticity
            else:
                total_elasticity = xr.ufuncs.maximum(total_elasticity, group_summed_elasticity)

    logger.info("Scaling the summed elasticity values from 0 to 1")
    total_elasticity = rescale(total_elasticity)
    dm_df = pd.DataFrame(dm_records)
    return total_elasticity, dm_df

if __name__ == "__main__":
    config = {"hash": "cedc9c8"}
    base_path = Path(__file__).parent / Path("../../data/processed")
    for hab in ["Aqu", "Ter"]:
        print("Processing type: ", hab)
        ecis, dm_df = calculate_ecis(config, hab, base_path)
        # Save result
        out_file = base_path / config["hash"] / f"ecological_connectivity_importance_score_{hab}"
        logger.info("Saving final raster to: %s", out_file)
        ecis.rio.to_raster(str(out_file) + ".tif", compress="zstd")
        dm_df.to_csv(str(out_file) + ".csv", index=False)
        if True:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plot_data = ecis.coarsen(x=10, y=10, boundary="trim").mean()
            plot_data.plot(
                ax=ax,
                # vmax=0.5,
                cmap="magma",
                cbar_kwargs={"label": f"Ecological Connectivity Importance Score\n{hab} species",
                            "shrink": 0.3},
            )
            ax.set_aspect("equal")
            ax.set_title("")
            ax.set_axis_off()
            # fig.tight_layout()
            fig.savefig(Path(__file__).parent / f"../../ecological_connectivity_importance_{hab}.png", dpi=300)
