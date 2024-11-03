using SparseArrays
using LinearAlgebra
using Graphs, GridGraphs
using BenchmarkTools
using CUDA

function fundamental_matrix(W::AbstractMatrix)
    Z = inv(I - W)
    return Z
end

# defining a random raster of permeability and associated grid graph
N = 50 # number of nodes
permeability = rand(N, N)
grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
W = Matrix(weights(grid)) # this matrix is sparse, but we make it dense for inversion

function objective(W)
    Z = fundamental_matrix(W)
    return sum(Z)
end

@time objective(W); # 0.319890 seconds (9 allocations: 96.607 MiB, 4.17% gc time)
cuW = CuArray(W);
@time objective(cuW); # 0.120776 seconds (434 allocations: 15.789 KiB)
# 5x speed up

using DifferentiationInterface
using Zygote
@time val, grad = value_and_gradient(objective, AutoZygote(), W); # 0.574153 seconds (44 allocations: 335.027 MiB)
@time val, grad = value_and_gradient(objective, AutoZygote(), cuW); #  0.258990 seconds (847 allocations: 27.477 KiB)
# 2x speed up