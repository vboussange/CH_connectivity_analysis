using SparseArrays
using LinearAlgebra
using Graphs, GridGraphs
using BenchmarkTools
using CUDA
using CUDA.CUSPARSE: AbstractCuSparseArray



function fundamental_matrix(W::AbstractMatrix)
    inv(I - W)
end

function mapnz(mat::M, f) where M <: AbstractSparseMatrix
    I, J, V = findnz(mat)
    return sparse(I, J, f.(V), size(mat)...)
end

function dense(sp_mat::M) where M <: AbstractSparseMatrix
    if M <: AbstractCuSparseArray
        return CuArray(sp_mat)
    else
        return Array(sp_mat)
    end
end

function rsp_distance(A::M, θ) where M <: AbstractSparseMatrix
    C = mapnz(A, x -> -log(x)) # cost matrix, here we assume the simplest case with well-adapted movements
    Prw = Diagonal(inv.(vec(sum(A, dims=2)))) * A # random walk probability
    W = Prw .* exp.(- θ .* C)
    Z = fundamental_matrix(dense(W))

    C̄ = Z * ((C .* W)*Z)
    
    C̄ = C̄ ./ Z

    C̄ = C̄ ./ (Z .+ eps(eltype(Z)))

    dˢ = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance(grid::GridGraph, θ::Real)
    A = sparse(weights(grid)) # affinities
    rsp_distance(A, θ)
end

# defining a random permeability matrix
len = 100 # height and width of the grid
permeability = rand(Float32, len, len)
grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)


θ = eltype(permeability)(0.01)
@time rsp_distance(grid, θ); # 9.437042 seconds (102 allocations: 2.988 GiB, 4.48% gc time)

## GPU test
Acu = sparse(CuArray(weights(grid))) # affinities
@time rsp_distance(Acu, θ); #   0.210289 seconds (5.61 k allocations: 866.430 KiB, 0.00% compilation time)
# 10X speed up, this is good!


using DifferentiationInterface
using Zygote

function total_distance(A)
    rsp_dist = rsp_distance(A, θ)
    return sum(rsp_dist)  # Sum of all coefficients in Z
end

len = 50 # height and width of the grid
permeability = rand(Float32, len, len)
grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
A = sparse(weights(grid)) # affinities
@time val, grad = value_and_gradient(total_distance, AutoZygote(), A); # 0.118560 seconds (480.48 k allocations: 46.673 MiB, 19.70% gc time)

# Takes for ever...
@time val, grad = value_and_gradient(total_distance, AutoZygote(), Acu) # 0.065373 seconds

# differentiation of weights

function objective(permeability)
    grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
    A = weights(grid)
    sum(A)
end

len = 100 # height and width of the grid
permeability = rand(Float32, len, len)
@time val, grad = value_and_gradient(objective, AutoZygote(), permeability) # 0.065373 seconds
