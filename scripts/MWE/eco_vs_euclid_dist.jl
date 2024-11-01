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
include(joinpath(@__DIR__, "../../src/utils.jl"))
include(joinpath(@__DIR__, "../../src/TraitsCH.jl"))
include(joinpath(@__DIR__, "../../src/grid.jl"))
include(joinpath(@__DIR__, "../../src/rsp_distance.jl"))
include(joinpath(@__DIR__, "../../src/euclid_distance.jl"))

dataset_path = joinpath(@__DIR__, "../../../data/compiled/GUILDS_EU_SP_buffer_dist=100km_resampling_1.nc")

sp_name = "Salmo trutta"
habitat_suitability = Raster(dataset_path; name=sp_name) / 100
habitat_suitability = replace_missing(habitat_suitability, 0.)
res = resolution(data_array) / 1000 #km

# only calculating for small window


affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(habitat_suitability))
g = Grid(habitat_suitability, affinity_matrix)
euclidean_distance = calculate_euclidean_distance(g, res)


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
