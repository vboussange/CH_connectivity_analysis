#=
Calcluating RSP distance and sensitivity of functional habitat to habitat quality
- Utilizing GPU
- Neglecting contibution of pixels to permeability
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
using ProgressMeter
include(joinpath(@__DIR__, "../../src/utils.jl"))
include(joinpath(@__DIR__, "../../src/TraitsCH.jl"))
include(joinpath(@__DIR__, "../../src/grid.jl"))
include(joinpath(@__DIR__, "../../src/rsp_distance.jl"))
include(joinpath(@__DIR__, "../../src/euclid_distance.jl"))

dataset_path = joinpath(@__DIR__, "../../../data/compiled/GUILDS_EU_SP_buffer_dist=100km_resampling_1.nc")


θ = Float32(0.01)
α = Float32(21)
traits = TraitsCH()
sp_name = "Salmo trutta"
data_array = Raster(dataset_path; name=sp_name) / 100
Plots.plot(data_array)



# D = get_D(traits, sp_name)
D_km = 10.
res = resolution(data_array) / 1000 #km

# width and height of window center
window_size = 100
buffer_size = ceil(Int, 3 * D_km / res)
step_size = window_size  # Step size for non-overlapping core windows
cut_off = 0.1
D = D_km / α

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
@showprogress for i in 1:(x_steps)
    for j in 1:(y_steps)
        x_start = (i-1) * step_size + 1
        y_start = (j-1) * step_size + 1
        # Extract the buffered window from the raw dataset
        hab_qual = replace_missing(data_array[x_start:(x_start+total_window_size),
                                    y_start:(y_start + total_window_size)], NaN)

        if !all(isnan.(hab_qual)) && any(hab_qual .> cut_off)
            # TODO: filter hab_qual .> cut_off
            affinity_matrix = ConScape.graph_matrix_from_raster(Matrix(hab_qual))

            # standard calculation
            grid = Grid(hab_qual, affinity_matrix)
            if nb_active(grid) > 1
                rsp_dist = rsp_distance_gpu(grid, θ);

                # calculate proximity
                K = exp.(-rsp_dist / D)

                q = CuArray([hab_qual[ij...] for ij in active_vertices_coordinate(grid)])

                # TODO: this could be simplified by not calculating for buffered values
                sensitivities_vec = gradient(q -> calculate_functional_habitat(q, K), q)[1] |> Vector # transferring back to cpu
                sensitivities = fill(NaN, height(grid), width(grid))
                [sensitivities[ij...] = sensitivities_vec[v] for (v, ij) in enumerate(active_vertices_coordinate(grid))]

                range = buffer_size:(buffer_size+window_size)
                output_array[x_start .+ range, y_start .+ range] = sensitivities[range, range]
            end
        end
    end
end

using Plots
Plots.plot(output_array)

