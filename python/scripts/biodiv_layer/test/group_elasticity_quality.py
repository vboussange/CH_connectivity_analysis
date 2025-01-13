"""
Calculating the elasticity of habitat quality with respect to permeability using Jaxscape.
TODO: need to verify that the batching and calculation are correct.

Printing D:
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU

import jax
import numpy as np
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.gridgraph import GridGraph
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.lcp_distance import LCPDistance

import equinox as eqx
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent / Path("../../../src/")))
from preprocessing import compile_group_suitability, CRS_CH
from processing import batch_run_calculation, padding, GROUP_INFO
from postprocessing import postprocess
import xarray as xr
import rioxarray
from copy import deepcopy
os.chdir(Path(__file__).parent)

if __name__ == "__main__":
    
    config = {"batch_size": 2**4, # pixels, actual batch size is batch_size**2
            "resolution": 100, # meters
            # percentage of the dispersal range, used to calculate landmarks
            # if the dispersal range is 10 pixels and the coarsening factor is 0.3, then the landmarks will be calculated every 2 pixels
            # each pixel should be involved by at least one landmark
            "coarsening_factor": 0.3,
            "dtype": "float32",
            }
    
    for group in GROUP_INFO:
        print("Computing elasticity for group:", group)
        distance = GROUP_INFO[group]
        try:

            output_path = Path("output") / group
            output_path.mkdir(parents=True, exist_ok=True)
            
            suitability_dataset = compile_group_suitability(group, 
                                                            config["resolution"])
            D_m = suitability_dataset.attrs["D_m"]
          
            ## Calculating meta parameters
            # dispersal in pixels
            D = np.array(D_m / config["resolution"], dtype=config["dtype"])
            assert D >= 1, "Dispersal range must be greater than 1 pixel."
            print(D)
            
        except Exception as e:
            print(f"Failed to compute elasticity for group {group}: {e}")
            continue