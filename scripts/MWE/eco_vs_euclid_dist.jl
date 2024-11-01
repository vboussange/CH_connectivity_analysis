# Regression between ecological distance and euclidean distance

cd(@__DIR__)
using GridGraphs, Graphs
using SparseArrays, DataFrames, CSV
using ProgressMeter
using DataFrames
using Printf
using Zygote
using NCDatasets
using Rasters
using ConScape
using CUDA, CUDA.CUSPARSE
using StatsBase # sample
include(joinpath(@__DIR__, "../../src/utils.jl"))
include(joinpath(@__DIR__, "../../src/TraitsCH.jl"))
include(joinpath(@__DIR__, "../../src/grid.jl"))
include(joinpath(@__DIR__, "../../src/rsp_distance.jl"))
include(joinpath(@__DIR__, "../../src/euclid_distance.jl"))

dataset_path = joinpath(@__DIR__, "../../../data/compiled/GUILDS_EU_SP_buffer_dist=100km_resampling_1.nc")

sp_name = "Salmo trutta"
habitat_suitability = Raster(dataset_path; name=sp_name) / 100
habitat_suitability = replace_missing(habitat_suitability, 0.)
res = resolution(habitat_suitability) / 1000 #km


# cropping for accelerating calcluations
raster_center = floor.(Int, size(habitat_suitability) ./ 2)
window_size = 100
myrange = -window_size÷2:window_size÷2
habitat_suitability = habitat_suitability[raster_center[1] .+ myrange, raster_center[2] .+ myrange]

# only calculating for small window

affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(habitat_suitability))

# standard calculation
grid = Grid(habitat_suitability, affinity_matrix)



# Euclidean distance calculation
# @time euclidean_dist = calculate_euclidean_distance(grid, res); # 0.340271 seconds (15 allocations: 305.755 MiB, 6.88% gc time)
@time euclidean_dist = calculate_euclidean_distance_gpu(grid, res); # 0.019106 seconds (497 allocations: 309.656 KiB)

θ = 0.01
# @time rsp_dist = rsp_distance(grid, θ); #  2.062014 seconds (107 allocations: 617.737 MiB, 5.38% gc time)
@time rsp_dist = rsp_distance_gpu(grid, θ); # 0.531023 seconds (4.96 k allocations: 1.220 MiB, 0.00% compilation time)

α =euclidean_dist[:] \ rsp_dist[:]


using Plots
rsp_dist = Array(rsp_dist)
euclidean_dist = Array(euclidean_dist)
rnd_idx = sample(1:length(euclidean_dist), 1000)
scatter(euclidean_dist[rnd_idx], rsp_dist[rnd_idx])
# Line of best fit
x_range = range(minimum(euclidean_dist), maximum(euclidean_dist), length=100)
plot!(x_range, α .* x_range, color=:red)
print("α: ", α)