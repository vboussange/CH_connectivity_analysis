import jax.numpy as jnp
import equinox as eqx
from jax import lax
from jaxscape.lcp_distance import LCPDistance
from jaxscape.euclidean_distance import EuclideanDistance

GROUP_INFO = {"Mammals": LCPDistance(),
              "Reptiles": LCPDistance(),
              "Amphibians": EuclideanDistance(),
              "Birds": EuclideanDistance(),
              "Fishes": LCPDistance(),
              "Plants": EuclideanDistance(),
              "Bryophytes": EuclideanDistance(),
              "Spiders": LCPDistance(),
              "Beetles": LCPDistance(),
              "Dragonflies": LCPDistance(),
              "Grasshoppers": LCPDistance(),
              "Butterflies": LCPDistance(),
              "Bees": LCPDistance(),
              "Fungi": EuclideanDistance(),
              "Gasteropods": LCPDistance(),
              "Lichens": EuclideanDistance()}
              

@eqx.filter_jit
def batch_run_calculation(batch_op, window_op, xy, fun, *args):
    raster_buffer = jnp.zeros((batch_op.total_window_size, batch_op.total_window_size))
    res = fun(*args)
    def scan_fn(raster_buffer, x):
        _xy, _rast = x
        raster_buffer = window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
        return raster_buffer, None
    raster_buffer, _ = lax.scan(scan_fn, raster_buffer, (xy, res))
    return raster_buffer

def padding(raster, buffer_size, window_size):
    """
    Pads the given raster array to ensure its dimensions are compatible with the
    specified window size, i.e. assert (raster.shape[i] - 2 * buffer_size) %
    window_size == 0
    """
    inner_height = raster.shape[0] - 2 * buffer_size
    inner_width = raster.shape[1] - 2 * buffer_size

    pad_height = (window_size - (inner_height % window_size)) % window_size
    pad_width = (window_size - (inner_width % window_size)) % window_size


    padded_raster = jnp.pad(
        raster,
        ((0,pad_height),(0,pad_width)),
        mode='constant'
    )
    return padded_raster

if __name__ == "__main__":
    from jaxscape.moving_window import WindowOperation

    def test_padding():
        raster = jnp.ones((2300, 3600))
        buffer_size = 50
        window_size = 3
        padded_raster = padding(raster, buffer_size, window_size)
        
        for i in range(2):
            assert (padded_raster.shape[i] - 2 * buffer_size) % window_size == 0
        
        # other test
        raster = jnp.ones((230, 360))
        buffer_size = 3
        window_size = 27
        padded_raster = padding(raster, buffer_size, window_size)

        for i in range(2):
            assert (padded_raster.shape[i] - 2 * buffer_size) % window_size == 0
        
    test_padding()
    
    # TODO: test batch_run_calculation