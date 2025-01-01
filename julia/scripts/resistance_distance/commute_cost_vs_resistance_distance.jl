include("InfiniteTemperature.jl")
using Graphs
using SimpleWeightedGraphs
using Statistics
using Graphs.LinAlg
function resistance_distance(g::AbstractGraph)
    laplacian = laplacian_matrix(g)

    L = Matrix(laplacian)
    L_inv = pinv(L)
    Linv_diag = diag(L_inv)

    R = @. Linv_diag + Linv_diag' - 2 * L_inv
    return R
end

## testing resistance distance
# g = SimpleWeightedGraph(4)
# add_edge!(g, 1, 2, 2)
# add_edge!(g, 2, 3, 4)
# add_edge!(g, 3, 4, 1)
# add_edge!(g, 1, 4, 3)
# R = resistance_distance(g)
# @assert R[1,3] ≈ 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1 + 1 / 3))


## testing equivalence between commute cost and resistance distance

## unweighted graph
g = Graph(4)
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)
add_edge!(g, 3, 4)
add_edge!(g, 1, 4)

A = adjacency_matrix(g)
C = deepcopy(A)
C.nzval .= 1 ./ C.nzval
# C = A .> 0 # this is for calculating commute time
P = A .* Diagonal(1 ./ sum(A, dims=2))

C̄ = CommuteCostFull(P, C)
R_resistance = resistance_distance(g)
@assert  cor(C̄[:], R_resistance[:]) ≈ 1 # true
@assert all(C̄ + C̄' ≈ sum(A) * R_resistance) # true


## weighted graph
g = SimpleWeightedGraph(4)
add_edge!(g, 1, 2, 2)
add_edge!(g, 2, 3, 4)
add_edge!(g, 3, 4, 1)
add_edge!(g, 1, 4, 3)


A = adjacency_matrix(g)
C = deepcopy(A)
C.nzval .= 1 ./ C.nzval 
# C = A .> 0 # this is for calculating commute time
P = A .* Diagonal(1 ./ sum(A, dims=2))

C̄ = CommuteCostFull(P, C)
R = resistance_distance(g)
@assert  cor((C̄ + C̄')[:], R_resistance[:]) ≈ 1 # true
@assert all(C̄ + C̄' ≈ sum(A) * R_resistance) # true

RC = C̄ + C̄'
@assert all(RC  ≈ 2 * ne(g) * R_resistance)
# @assert all(RC  ≈ sum(1 ./ A.nzval) * R_resistance)

# THIS is FALSE 
# TODO: to revise based on Matlab code of Marco Saerens