from utils_raster import crop_raster
from swissTLMRegio import MasksDataset, get_CH_border


def postprocess(output):
    """We crop the output raster to the extent of Switzerland"""
    switzerland_boundary = get_CH_border()
    output = crop_raster(output, switzerland_boundary)
    return output