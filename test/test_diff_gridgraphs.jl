using GridGraphs
using NCDatasets, Rasters
using Graphs
using Zygote
using ForwardDiff
using BenchmarkTools

dataset_path = joinpath(@__DIR__, "habitat_suitability.nc")

sp_name = "Salmo trutta"
habitat_suitability = Raster(dataset_path; name=sp_name) / 100
habitat_suitability = replace_missing(habitat_suitability, 0.)
g = GridGraph(habitat_suitability)

function dummy(habitat_suitability)
    g = GridGraph(habitat_suitability)
    K = adjacency_matrix(g)
    sum(habitat_suitability[:] .* (K * habitat_suitability[:]))
end

dummy(habitat_suitability)

sensitivities_vec = gradient(q -> dummy(q), habitat_suitability)[1]

@time sensitivities = ForwardDiff.gradient(q -> dummy(q), habitat_suitability)
using Plots
plot(sensitivities)
plot(habitat_suitability)