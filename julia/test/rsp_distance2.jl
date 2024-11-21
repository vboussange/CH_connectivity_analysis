import ConScape
include("../src/grid.jl")
include("../src/rsp_distance.jl")

θ = 1.
A = sparse([
    0 1.2 1.2 0 0 0;
    1.2 0 0 1.2 0 0;
    1.2 0 0 0 1.2 0;
    0 1.5 0 0 0 1.5;
    0 0 1.5 0 0 1.5;
    0 0 0 1.5 1.5 0])

rsp_distance(A, θ)
