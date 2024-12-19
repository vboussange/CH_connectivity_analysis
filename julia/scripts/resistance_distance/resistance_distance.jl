using Graphs, LinearAlgebra, SimpleWeightedGraphs

"""
    resistance_distance(g::Graph, u::Int, v::Int)

Calculates the resistance distance between nodes `u` and `v` in graph `g`.

# https://github.com/networkx/networkx/blob/main/networkx/algorithms/tests/test_distance_measures.py 

--> careful whether edges represent cost or proximity, this is treated in networkx
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/distance_measures.html#resistance_distance 
"""
function resistance_distance(g::AbstractGraph, u::Int, v::Int)
    laplacian = laplacian_matrix(g)

    L = Matrix(laplacian)
    L_inv = pinv(L)

    r = L_inv[u, u] + L_inv[v, v] - 2 * L_inv[u, v]
    return r
end

function resistance_distance(g::AbstractGraph)
    laplacian = laplacian_matrix(g)

    L = Matrix(laplacian)
    L_inv = pinv(L)
    Linv_diag = diag(L_inv)

    R = @. Linv_diag + Linv_diag' - L_inv' - L_inv
    return R
end

# test
g = SimpleWeightedGraph(4)
add_edge!(g, 1, 2, 2)
add_edge!(g, 2, 3, 4)
add_edge!(g, 3, 4, 1)
add_edge!(g, 1, 4, 3)
R = resistance_distance(g)
@assert R[1,3] ≈ 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1 + 1 / 3))
Ruv = resistance_distance(g, 1, 3)
@assert Ruv ≈ 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1 + 1 / 3))