include("InfiniteTemperature.jl")
using Graphs
using SimpleWeightedGraphs
function resistance_distance(g::AbstractGraph)
    laplacian = laplacian_matrix(g)

    L = Matrix(laplacian)
    L_inv = pinv(L)
    Linv_diag = diag(L_inv)

    R = @. Linv_diag + Linv_diag' - L_inv' - L_inv
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


P = adjacency_matrix(g) ./ sum(adjacency_matrix(g), dims=2)
C = adjacency_matrix(g)

C̄ = CommuteCostFull(P, C)
R_resistance = resistance_distance(g)
RC = C̄ + C̄'
# https://en.wikipedia.org/wiki/Resistance_distance#Relationship_to_random_walks
@assert all(RC  ≈ 2 * ne(g) * R_resistance)
# This is TRUE

## weighted graph
g = SimpleWeightedGraph(4)
add_edge!(g, 1, 2, 2)
add_edge!(g, 2, 3, 4)
add_edge!(g, 3, 4, 1)
add_edge!(g, 1, 4, 3)


A = adjacency_matrix(g)
P = deepcopy(A)
P.nzval .= 1 ./ A.nzval
P= P ./ sum(P, dims=2)
C = adjacency_matrix(Graph(g))

C̄ = CommuteCostFull(P, C)
R = resistance_distance(g)

RC = C̄ + C̄'

@assert all(RC  ≈ 2 * ne(g) * R_resistance)
@assert all(RC  ≈ sum(1 ./ A.nzval) * R_resistance)

# THIS is FALSE 