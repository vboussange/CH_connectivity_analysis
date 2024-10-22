using SparseArrays
using Zygote
using LinearAlgebra
using Graphs, GridGraphs
using BenchmarkTools

function fundamental_matrix(W::SparseMatrixCSC, targetnodes::Vector{Int})
    Z = (I - W) \ Matrix(sparse(targetnodes, 1:length(targetnodes), 1.0, size(W, 1), length(targetnodes)))
    return Z
end


# defining a random raster of permeability and associated grid graph
N = 10 # number of nodes
permeability = rand(N, N) / 4
grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
W = adjacency_matrix(grid)

# Define the target nodes 
targetnodes = collect(vertices(grid))

# Define the objective function: sum of the coefficients of the fundamental matrix
function objective(W)
    Z = fundamental_matrix(W, targetnodes)
    return sum(Z)  # Sum of all coefficients in Z
end

# Reference value
# 7.448 ms (37 allocations: 371.28 KiB)
@btime objective($W)

# Calculation of the gradient (sensitivity) of the sum of Z with respect to the transition matrix W
# 16.018 ms (93 allocations: 894.89 KiB)
@btime gradient($objective, $W)

