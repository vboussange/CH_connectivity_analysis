using NCDatasets, Rasters
import ConScape
using Plots
using Test
include("../src/grid.jl")
include("../src/rsp_distance.jl")



dataset_path = joinpath(@__DIR__, "habitat_suitability.nc")

sp_name = "Salmo trutta"
habitat_suitability = Raster(dataset_path; name=sp_name) / 100
habitat_suitability = replace_missing(habitat_suitability, 0.)
plot(habitat_suitability)

θ = 0.01
# CONSCAPE calculation
affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(habitat_suitability))

grid = ConScape.Grid(size(habitat_suitability)...,
                        affinities=affinity_matrix,
                        source_qualities=Matrix(habitat_suitability),
                        target_qualities=sparse(habitat_suitability),
                        costs=ConScape.mapnz(x -> -log(x), affinity_matrix))

grsp = ConScape.GridRSP(grid; θ)
dist_conscape =  ConScape.expected_cost(grsp)

# Custom calculation
grid = Grid(habitat_suitability, affinity_matrix)
dist = ecological_distance(grid)
@test all(dist .≈ dist_conscape)

# calculation of proximity and functional habitat
D = 1.
K = exp.(-dist / D)
q = [habitat_suitability[v] for v in list_active_vertices(grid)]
calculate_functional_habitat(q, K)

# differentiating functional habitat
# TODO: Seems to work but is very very slow
function calculate_functional_habitat(habitat_suitability)

    affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(habitat_suitability))

    # Custom calculation
    grid = Grid(habitat_suitability, affinity_matrix, true)
    dist = ecological_distance(grid)

    # calculation of pro
    D = 1.
    K = exp.(-dist / D)
    q = [habitat_suitability[v] for v in list_active_vertices(grid)]
    calculate_functional_habitat(q, K)
end

calculate_functional_habitat(habitat_suitability)

using ForwardDiff
sensitivities = ForwardDiff.gradient(q -> calculate_functional_habitat(q), habitat_suitability)
