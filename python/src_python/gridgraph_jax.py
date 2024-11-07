import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

def create_grid_graph(permeability_raster):
    """
    Create a differentiable grid graph based on a permeability raster in JAX.

    Args:
        permeability_raster (jnp.ndarray): A 2D array representing permeability values.

    Returns:
        BCOO: A sparse matrix representing the connectivity in the grid graph, 
              where the values are based on the permeability raster.
    """
    # Get shape of raster
    nrows, ncols = permeability_raster.shape
    num_nodes = nrows * ncols

    # Index arrays for each position
    row_indices = jnp.repeat(jnp.arange(nrows), ncols)
    col_indices = jnp.tile(jnp.arange(ncols), nrows)

    # Flattened 2D grid indices
    indices = jnp.arange(num_nodes).reshape(nrows, ncols)

    # Neighboring indices in the grid
    neighbors = [
        (1, 0),  # down
        (-1, 0),  # up
        (0, 1),  # right
        (0, -1),  # left
    ]

    # Collect data for sparse matrix
    rows, cols, values = [], [], []

    for dr, dc in neighbors:
        # Compute new indices with boundary adjustments using where
        new_row_indices = row_indices + dr
        new_col_indices = col_indices + dc

        # Replace out-of-bound indices with the current position indices (invalid moves will have zero weight)
        valid_new_row_indices = jnp.where((new_row_indices >= 0) & (new_row_indices < nrows), new_row_indices, row_indices)
        valid_new_col_indices = jnp.where((new_col_indices >= 0) & (new_col_indices < ncols), new_col_indices, col_indices)

        # Flattened indices for current and neighboring cells
        current_indices = indices[row_indices, col_indices]
        neighbor_indices = indices[valid_new_row_indices, valid_new_col_indices]

        # Connectivity values based on permeability, using where to set invalid moves to zero
        value = jnp.where(
            (new_row_indices >= 0) & (new_row_indices < nrows) & (new_col_indices >= 0) & (new_col_indices < ncols),
            permeability_raster[row_indices, col_indices],
            0.0
        )

        # Append values and indices
        values.extend(value)
        rows.extend(current_indices)
        cols.extend(neighbor_indices)

    # Stack results into arrays for BCOO format
    data = jnp.array(values)
    row_col_indices = jnp.vstack([jnp.array(rows), jnp.array(cols)]).T

    # Create sparse matrix in BCOO format
    graph = BCOO((data, row_col_indices), shape=(num_nodes, num_nodes))

    return graph



import jax.numpy as jnp
from jax import grad, jit

@jit
def flow_efficiency(adjacency_matrix):
    """
    Computes the flow efficiency as the sum of all edge weights in the adjacency matrix.
    
    Args:
        adjacency_matrix (jnp.ndarray): The adjacency matrix of the grid graph.
    
    Returns:
        float: The flow efficiency metric.
    """
    # Sum of all edge weights (undirected graph, so divide by 2 to avoid double-counting)
    return adjacency_matrix.sum() / 2.0

@jit
def objective_function(permeability_raster):
    """
    Computes the objective function based on permeability.
    
    Args:
        permeability_raster (jnp.ndarray): A 2D array where each value represents the
                                           permeability at that point.
    
    Returns:
        float: The objective value representing flow efficiency.
    """
    # Build grid graph adjacency matrix from the permeability raster
    adjacency_matrix = create_grid_graph(permeability_raster)
    
    # Compute flow efficiency
    return flow_efficiency(adjacency_matrix)

# Calculate the gradient of the objective function w.r.t. the permeability raster
grad_objective = grad(objective_function)


# Example permeability raster
import jax.random as jr
key = jr.PRNGKey(0)  # Random seed is explicit in JAX
permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability

adj = create_grid_graph(permeability_raster)

# Compute gradient of flow efficiency with respect to permeability
permeability_gradient = grad_objective(permeability_raster)




# matrix solve
import lineax as lx
matrix_key, vector_key = jr.split(jr.PRNGKey(0))
matrix = jr.normal(matrix_key, (10, 8))
vector = jr.normal(vector_key, (10,))
operator = lx.MatrixLinearOperator(matrix)
solution = lx.linear_solve(operator, vector, solver=lx.QR())