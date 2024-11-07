using SparseArrays
using Zygote
using LinearAlgebra
using Graphs, GridGraphs
using BenchmarkTools
using CUDA

function fundamental_matrix(W::AbstractMatrix)
    Z = inv(I - W)
    return Z
end

# defining a random raster of permeability and associated grid graph
N = 100 # number of nodes
permeability = rand(N, N) / 4
grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
A = Matrix{Float32}(weights(grid)) # affinities

# Reference value
fundamental_matrix(A)
@time fundamental_matrix(A);
# 5.903434 seconds (8 allocations: 765.457 MiB, 0.61% gc time)


Acu = CuArray(A)
fundamental_matrix(Acu)
@time fundamental_matrix(Acu);
# 0.200090 seconds (319 allocations: 12.102 KiB)