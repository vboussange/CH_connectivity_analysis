using SparseArrays
using LinearAlgebra
using Graphs, GridGraphs
using BenchmarkTools
using CUDA
using CUDA.CUSPARSE: AbstractCuSparseArray


function fundamental_matrix(W::AbstractArray)
    inv(I - W)
end

function mapnz(mat::M, f) where M <: AbstractSparseArray
    I, J, V = findnz(mat)
    return sparse(I, J, f.(V), size(mat)...)
end

function dense(sp_mat::M) where M <: AbstractSparseArray
    if M <: AbstractCuSparseArray
        return CuArray(sp_mat)
    else
        return Array(sp_mat)
    end
end


logpos(a) = a > 0 ? log(a) : zero(a)

function rsp_distance(A::M, θ::T) where {T, M <: AbstractSparseArray}
    C = - logpos.(A)
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

using DifferentiationInterface
using Zygote

function total_distance(A)
    rsp_dist = rsp_distance(A, θ)
    return sum(rsp_dist)  # Sum of all coefficients in Z
end

len = 10 # height and width of the grid
permeability = rand(Float32, len, len)
const θ = eltype(permeability)(0.01)

grid = GridGraph(permeability, directions=ROOK_DIRECTIONS)
A = sparse(weights(grid)) # affinities
@time total_distance(A) # 0.338271 seconds (167 allocations: 192.544 MiB, 7.30% gc time, 2.53% compilation time)
@time val, grad = value_and_gradient(total_distance, AutoZygote(), A); # 0.118560 seconds (480.48 k allocations: 46.673 MiB, 19.70% gc time)

Acu = sparse(CuArray(weights(grid)))
@time total_distance(Acu); # 0.041128 seconds (5.22 k allocations: 326.703 KiB, 0.01% compilation time)
# Takes for ever...
@time val, grad = value_and_gradient(total_distance, AutoZygote(), Acu)


# trying to isolate the problem
function isolate(A::M) where M <: AbstractSparseArray
    # C = mapnz(A, x -> -log(x)) # cost matrix, here we assume the simplest case with well-adapted movements
    C = - logpos.(A)
    # Prw = Diagonal(inv.(vec(sum(A, dims=2)))) * A # random walk probability
    # W = Prw .* exp.(- θ .* C)
    # Z = fundamental_matrix(dense(W))

    # C̄ = Z * ((C .* W)*Z)
    
    # C̄ = C̄ ./ Z

    # C̄ = C̄ ./ (Z .+ eps(eltype(Z)))

    # dˢ = diag(C̄)
    # C̄ = C̄ .- dˢ'
    # return C̄
    return sum(C)
end
@time isolate(Acu); # 0.041128 seconds (5.22 k allocations: 326.703 KiB, 0.01% compilation time)
# Takes for ever...
@btime val, grad = value_and_gradient(isolate, AutoZygote(), Acu)

using Enzyme
@time val, grad = value_and_gradient(isolate, AutoEnzyme(), Acu)

@btime val, grad = value_and_gradient(isolate, AutoZygote(), A)


# seems that the problem comes from mapnz
@btime log.($Acu); # 27.378 μs (107 allocations: 3.61 KiB)
@btime mapnz($Acu, log); # 1.123 ms (1848 allocations: 52.81 KiB)
@btime begin #this seems to be a working implementation
    C = - ifelse.(Acu .> zero(eltype(Acu)), log.(Acu), Acu)
end


