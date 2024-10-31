#=
Run zonation algorithm on each guild separately
=#

cd(@__DIR__)
using GridGraphs, Graphs
using SparseArrays, DataFrames, CSV
using ProgressMeter
using DataFrames
using JLD2
using Printf
using PythonPlot
using Zygote
using NCDatasets
using Rasters
using PythonCall
include(joinpath(@__DIR__, "../../src/TraitsCH.jl"))

dataset_path = joinpath(@__DIR__, "../../../data/GUILDS_EU_SP/GUILDS_EU_SP_buffer_dist=100km_resampling_1.nc")

function id_to_grid_coordinate_list(g::GridGraph)
    [index_to_coord(g, v) for v in vertices(g) if vertex_active(g, v)]
end

function calculate_euclidean_distance(g::GridGraph, res)
    coordinate_list = id_to_grid_coordinate_list(g)
    euclidean_distance = [hypot(xy_i[1] - xy_j[1], xy_i[2] - xy_j[2]) for xy_i in coordinate_list, xy_j in coordinate_list]
    return euclidean_distance * res
end
    

function calculate_functional_habitat(q, K)
    return sum(q .* (K * q))
end

function resolution(ras::Raster)
    lon = lookup(ras, X) 
    return lon[2] - lon[1]
end

sp_name = "Salmo trutta"
data_array = Raster(dataset_path; name=sp_name) / 100
traits = TraitsCH()
# D = get_D(traits, sp_name)
D = 1.
res = resolution(data_array) / 1000 #km

# width and height of window center
window_size = 40
buffer_size = ceil(Int, 3 * D / res)
step_size = window_size  # Step size for non-overlapping core windows
cut_off = 0.1

# calculate rolling window
total_window_size = window_size + 2 * buffer_size

# Number of steps (how many non-overlapping windows can be extracted)
width_raster = size(data_array, 1)
height_raster = size(data_array, 2)
x_steps = (width_raster - buffer_size * 2) ÷ window_size
y_steps = (height_raster - buffer_size * 2) ÷ window_size

output_array = copy(data_array)
output_array .= NaN
# output_array = fill(NaN, width_raster, height_raster)
# Now we iterate over the buffered windows
for i in 1:(x_steps-1)
    for j in 1:(y_steps-1)
        x_start = i * step_size
        y_start = j * step_size
        # Extract the buffered window from the raw dataset
        hab_qual = replace_missing(data_array[x_start:(x_start+total_window_size),
                                    y_start:(y_start + total_window_size)], NaN)

        if !all(isnan.(hab_qual)) && any(hab_qual .> cut_off)
            g = GridGraph(hab_qual; vertex_activities = hab_qual .> cut_off)
            euclidean_distance = calculate_euclidean_distance(g, res)

            # calculate proximity
            K = exp.(-euclidean_distance / D)

            q = [hab_qual[ij...] for ij in id_to_grid_coordinate_list(g)]

            # TODO: this could be simplified by not calculating for buffered values
            sensitivities_vec = gradient(q -> calculate_functional_habitat(q, K), q)[1]
            sensitivities = fill(NaN, height(g), width(g))
            [sensitivities[ij...] = sensitivities_vec[v] for (v, ij) in enumerate(id_to_grid_coordinate_list(g))]

            range = buffer_size:(buffer_size+window_size)
            output_array[x_start .+ range, y_start .+ range] = sensitivities[range, range]
        end
    end
end

using Plots
Plots.plot(output_array)
