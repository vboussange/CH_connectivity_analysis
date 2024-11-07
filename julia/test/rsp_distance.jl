using NCDatasets, Rasters
import ConScape
using Plots
using Test
using DifferentiationInterface
using BenchmarkTools
using DelimitedFiles
include("../src/grid.jl")
include("../src/rsp_distance.jl")

function get_canvas(g::ConScape.Grid, values)
    canvas = fill(NaN, g.nrows, g.ncols)
    for (i,v) in enumerate(values)
        canvas[g.id_to_grid_coordinate_list[i]] = v
    end
    return canvas
end

dataset_path = joinpath(@__DIR__, "habitat_suitability.nc")

sp_name = "Salmo trutta"
habitat_suitability = Raster(dataset_path; name=sp_name) / 100
habitat_suitability = replace_missing(habitat_suitability, 0.) |> Matrix
plot(habitat_suitability)

θ = 0.01

## CONSCAPE ecological distance calculation
affinity_matrix = ConScape.graph_matrix_from_raster(habitat_suitability, neighbors=ConScape.N4)

grid = ConScape.Grid(size(habitat_suitability)...,
                        affinities=affinity_matrix,
                        source_qualities=habitat_suitability,
                        target_qualities=sparse(habitat_suitability),
                        costs=ConScape.mapnz(x -> -log(x), affinity_matrix))

dist_conscape = @btime begin 
    grsp = ConScape.GridRSP(grid; θ)
    ConScape.expected_cost(grsp)
end # 4.250 ms (602 allocations: 6.74 MiB)
vertex = 5
coord = grid.id_to_grid_coordinate_list[vertex]
dist_to_vertex = get_canvas(grid, dist_conscape[:, 5])
writedlm("habitat_suitability.csv", habitat_suitability, ',')
writedlm("conscape_rsp_distance_to_i=$(coord[1])_j=$(coord[2]).csv", dist_to_vertex, ',')

# ## ./src ecological distance calculation
# dist = @btime rsp_distance(grid, θ) # 2.967 ms (72 allocations: 5.19 MiB)
# @test all(dist .≈ dist_conscape)

## ./src ecological distance calculation from affinity
grid = Grid(habitat_suitability, affinity_matrix)
active_vertices = list_active_vertices(grid)
A_pruned = affinity_matrix[active_vertices, 
                            active_vertices]
writedlm("$(sp_name)_conscape_affinity_matrix.csv", A_pruned, ',')
dist_2 = @btime rsp_distance(A_pruned, θ) # 2.825 ms (54 allocations: 7.30 MiB)
@test all(dist_2 .≈ dist_conscape)
sum(dist_2)

active_vertices = list_active_vertices(grid)
A_pruned = affinity_matrix[active_vertices, 
                            active_vertices] .|> Float32
dist_2 = @btime rsp_distance(A_pruned, θ) # 2.825 ms (54 allocations: 7.30 MiB)
@test all(dist_2 .≈ dist_conscape)
sum(dist_2)

## ./src ecological distance calculation from affinity
A_pruned = affinity_matrix[active_vertices, 
                            active_vertices]
dist_2 = @btime ecological_distance(A_pruned, θ) # 2.825 ms (54 allocations: 7.30 MiB)
@test all(dist_2 .≈ dist_conscape)

## ./src ecological distance calculation from affinity
dist_3 = @btime ecological_distance_sparse(affinity_matrix, θ, active_vertices) # 102.645 ms (229 allocations: 129.96 MiB)
@test all(dist_3[active_vertices, active_vertices] .≈ dist_conscape)

## ./src ecological distance calculation from affinity
dist_4 = @btime ecological_distance(affinity_matrix, θ, active_vertices) # 43.743 ms (149 allocations: 109.19 MiB)
@test all(dist_4[active_vertices, active_vertices] .≈ dist_conscape)

## test of differentiability w.r.t affinities
using Zygote
using ForwardDiff
# value_and_gradient(A -> sum(ecological_distance(A, θ)), AutoEnzyme(), A_pruned)
@btime sum(ecological_distance(A_pruned, θ)) # 2.913 ms (55 allocations: 7.30 MiB)

@time val, grad = value_and_gradient(A -> sum(ecological_distance(A, θ)), AutoZygote(), Matrix(A_pruned)) # 0.065373 seconds
# val, grad = value_and_gradient(A -> sum(ecological_distance(A, θ)), AutoForwardDiff(), A_pruned) # runs for ever
# val, grad = value_and_gradient(A -> sum(map(x -> ifelse(x > 0, -log(x), zero(eltype(A))), A)), AutoForwardDiff(), A_pruned) # returns (5.0, [2.0, 4.0]) with Enzyme.jl
@time val, grad2 = value_and_gradient(A -> sum(ecological_distance(A, θ, active_vertices)), AutoZygote(), affinity_matrix) # 1.445355 seconds (138.35 k allocations: 662.671 MiB, 2.38% gc time, 7.48% compilation time)
@time val, grad3 = value_and_gradient(A -> sum(ecological_distance_sparse(A, θ, active_vertices)), AutoZygote(), affinity_matrix) # 1.445355 seconds (138.35 k allocations: 662.671 MiB, 2.38% gc time, 7.48% compilation time)

@test all(grad2[active_vertices, active_vertices] .≈ grad)


## calculation of proximity and functional habitat
D = 1.
K = exp.(-dist / D)
q = [habitat_suitability[v] for v in active_vertices]
calculate_functional_habitat(q, K)

## differentiating functional habitat
# TODO: Seems to work but is very very slow
function calculate_functional_habitat(habitat_suitability)
    affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(habitat_suitability))

    # Custom calculation
    grid = Grid(habitat_suitability, affinity_matrix, true)
    dist = ecological_distance(grid)

    # calculation of pro
    D = 1.
    K = exp.(-dist / D)
    q = [habitat_suitability[v] for v in active_vertices]
    calculate_functional_habitat(q, K)
end
calculate_functional_habitat(habitat_suitability)