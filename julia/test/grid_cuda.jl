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
range = -window_size÷2:window_size÷2
habitat_suitability = habitat_suitability[raster_center[1] .+ range, raster_center[2] .+ range]

# only calculating for small window

affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(habitat_suitability))

# # CUDA calculation
using Test

# test 2
θ = Float32(0.01)
cuaffinity_matrix = CuSparseMatrixCSC{Float32}(affinities(grid)[active_vertices, 
                                                active_vertices])
active_vertices = list_active_vertices(grid)
@time ecological_dis_gpu = ecological_distance_gpu(cuaffinity_matrix, θ)
#   0.042320 seconds (5.38 k allocations: 499.156 KiB, 0.01% compilation time)
ecological_dis_gpu = Matrix(ecological_dis_gpu)
@test all(isapprox.(ecological_dis_gpu, ecological_dis, rtol=1e-2))
# Massive speedup!



# differentiability / this does not work
using DifferentiationInterface, Zygote
function myfun(cuaffinity_matrix)
    return sum(ecological_distance_gpu(cuaffinity_matrix, θ))
end
sum(ecological_distance_gpu(cuaffinity_matrix, θ))
@time val, grad = value_and_gradient(myfun, AutoZygote(), cuaffinity_matrix) # 0.065373 seconds



# This fails
# θ = Float32(0.01)
# cuaffinity_matrix = CuSparseMatrixCSC{Float32}(affinities(grid))
# active_vertices = list_active_vertices(grid)
# @time ecological_dis_gpu = ecological_distance_gpu(cuaffinity_matrix, θ, active_vertices)
# #   0.027061 seconds (4.60 k allocations: 26.188 MiB, 0.02% compilation time)
# ecological_dis_gpu = Matrix(ecological_dis_gpu[active_vertices, active_vertices])
# @test all(isapprox.(ecological_dis_gpu, ecological_dis, rtol=1e-1))
