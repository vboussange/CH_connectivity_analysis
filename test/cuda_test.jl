using SparseArrays
using Zygote
using LinearAlgebra
using Graphs, GridGraphs
using BenchmarkTools
using CUDA

function fundamental_matrix(W::SparseMatrixCSC, targetnodes::Vector{Int})
    Z = (I - W) \ Matrix(sparse(targetnodes, 1:length(targetnodes), one(eltype(W)), size(W, 1), length(targetnodes)))
    return Z
end

# defining a random raster of permeability and associated grid graph
N = 20 # number of nodes
permeability = rand(N, N) / 4
grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
W = adjacency_matrix(grid) .|> Float32

targetnodes = collect(vertices(grid))

# Reference value
@btime fundamental_matrix($W, $targetnodes)
# 726.336 ms (41 allocations: 4.25 MiB)


Wcu = CuArray(W)
function fundamental_matrix_gpu(W, targetnodes)
    A = (I - W)
    Z = A \ CuArray(Matrix(sparse(targetnodes, 1:length(targetnodes), one(eltype(W)), size(W, 1), length(targetnodes))))
    return Z
end

@btime fundamental_matrix_gpu($Wcu, $targetnodes)
# 1.318 ms (233 allocations: 652.25 KiB)