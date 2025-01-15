import rioxarray
import xarray as xr
import numpy as np
from scipy.ndimage import binary_dilation, grey_dilation
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / Path("../../../src/")))
from TraitsCH import TraitsCH
from NSDM import NSDM
from utils_raster import crop_raster
from masks import get_CH_border
raster_NSDM = NSDM().load_raster("Squalius cephalus").squeeze()
# raster_NSDM.plot()
# data_mask = raster_NSDM.isnull().data

# structuring_element = np.ones((3, 3))  # Define a 3x3 structuring element
# dilated_mask = grey_dilation(raster_NSDM, structure=structuring_element, iterations=10)
# plt.imshow(dilated_mask)
# dilated_data = raster_NSDM.where(~data_mask, dilated_mask)
# dilated_data.plot()
# # dilated_ds = raster_NSDM.copy()
# # dilated_ds.data = dilated_data

# # dilated_ds.rio.to_raster("dilated_dataset.tif")


# def fillna_dilation(
#     da: xr.DataArray,
#     structure=None,
#     iterations: int = 1,
#     fill_value=-np.inf
# ) -> xr.DataArray:
#     """
#     Fill NaN values using a grey-level dilation approach.

#     Parameters
#     ----------
#     da : xr.DataArray
#         Input data with NaNs to fill.
#     structure : np.ndarray, optional
#         Structuring element passed to `grey_dilation`. 
#         If None, uses a 3x3 (2D) or 3x1... "cube" for ND arrays.
#     iterations : int
#         Number of times to apply the dilation. More iterations
#         can fill bigger holes.
#     fill_value : float
#         Temporary sentinel value to replace NaNs. Should be something
#         'lower' than all valid data if we're using dilation (which
#         fills with the maximum among neighbors).

#     Returns
#     -------
#     xr.DataArray
#         A new DataArray with NaNs filled (where possible).
#     """
#     # 1. Copy to avoid modifying original
#     filled_da = da.copy()

#     # 2. Replace NaNs with a sentinel value
#     data_np = filled_da.values
#     nan_mask = np.isnan(data_np)
#     data_np[nan_mask] = fill_value

#     # If user didn’t specify a structuring element, build one that matches dimensionality:
#     if structure is None:
#         # Example: a “3x3” for 2D, “3x3x3” for 3D, etc.
#         # One-liner to create a size-3 “cube” across all dims
#         structure = np.ones([3]*data_np.ndim, dtype=bool)

#     # 3. Iteratively apply grey_dilation
#     for _ in range(iterations):
#         data_np = grey_dilation(data_np, footprint=structure)
#         filled_da.values = data_np

#     # 4. Where the original array had NaNs but we were able to fill them,
#     #    they now have "max-of-neighbors" values. In principle, if the “hole”
#     #    is bigger than the neighborhood, you might need more iterations or
#     #    a different approach.

#     return filled_da

# raster_NSDM.plot()
# raster_NSDM.interpolate_na(dim="x").plot()

# raster_NSDM.plot()
# filled_da = fillna_dilation(raster_NSDM, structure=np.ones((3, 3)), iterations=500)
# filled_da.plot()


# import numpy as np
# import xarray as xr
# from scipy import ndimage

def fill_na_with_nearest(da: xr.DataArray) -> xr.DataArray:
    """
    Fill NA values by assigning each NA cell the value of its nearest non-NA cell
    in Euclidean distance, preserving original non-NA cells.
    """
    data_np = da.values
    mask = np.isnan(data_np)
    
    # distance_transform_edt gives us for each point in `mask`:
    #   - the distance to the nearest False (i.e., nearest valid)
    #   - the indices of that nearest valid
    # So first invert mask to mark non-NA as False, NA as True:
    dist, (inds_y, inds_x, *other_inds) = ndimage.distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True
    )
    # NOTE: If da has more than 2 dimensions, you get multiple index arrays.
    # E.g. for 3D, you'll get (inds_z, inds_y, inds_x).
    # We'll just keep a conceptual example for 2D or shape out as needed.
    
    filled_np = data_np.copy()
    # Fill the NA cells using the nearest valid cell’s value
    filled_np[mask] = data_np[inds_y[mask], inds_x[mask]]
    
    # Return a new DataArray with the same coords, etc.
    filled_da = da.copy()
    filled_da.values = filled_np
    return filled_da



raster_NSDM.plot()
filled_raster = fill_na_with_nearest(raster_NSDM)
filled_raster.plot()

switzerland_buffer = get_CH_border().buffer(10000)
crop_raster(raster_NSDM, switzerland_buffer)