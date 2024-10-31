#=
Run zonation algorithm on each guild separately
=#

cd(@__DIR__)
using PythonCall
include("../../src/landscape.jl") # orders matters!
using GridGraphs, Graphs
using SparseArrays, DataFrames, CSV
using ProgressMeter
using DataFrames
using JLD2
using Printf
using PythonPlot
using Zygote
np = pyimport("numpy")
sys = pyimport("sys")
sys.path.append(joinpath(@__DIR__, "../preprocessing/"))
TraitsCH = pyimport("TraitsCH").TraitsCH
Preprocessing = pyimport("preprocessing_habitat_suitability_GUILDES_EU_SP")
plt = pyimport("matplotlib.pyplot")

dataset_path = joinpath(@__DIR__, "../../../data/GUILDS_EU_SP/GUILDS_EU_SP_buffer_dist=100km_resampling_1.nc")

function id_to_grid_coordinate_list(g::GridGraph)
    [index_to_coord(g, v) for v in vertices(g) if vertex_active(g, v)]
end

function calculate_euclidean_distance(g::GridGraph, res)
    coordinate_list = id_to_grid_coordinate_list(g)
    euclidean_distance = [hypot(xy_i[1] - xy_j[1], xy_i[2] - xy_j[2]) for xy_i in coordinate_list, xy_j in coordinate_list]
    return euclidean_distance * res
end

function add!(dataset, layer_name, layer)
    dataset[layer_name] = (dataset.dims, layer)
end
    

function calculate_functional_habitat(q, K)
    return sum(q .* (K * q))
end


dataset = load_xr_dataset(dataset_path)
trait_dataset = TraitsCH()
sp_name = "Salmo trutta"
# D = pyconvert(Float64, trait_dataset.get_D(sp_name))
D = 1.
res = pyconvert(Float64, Preprocessing.calculate_resolution(dataset)[1]) / 1000 #km


# width and height of window center
window_size = 40
buffer_size = ceil(Int, 3 * D / res)
step_size = window_size  # Step size for non-overlapping core windows
cut_off = 0.1

# calculate rolling window
total_window_size = window_size + 2 * buffer_size

# Number of steps (how many non-overlapping windows can be extracted)
width_raster = pyconvert(Int, dataset.sizes['x'])
height_raster = pyconvert(Int, dataset.sizes['y'])
x_steps = (width_raster - buffer_size * 2) ÷ window_size
y_steps = (height_raster - buffer_size * 2) ÷ window_size


data_array = dataset[sp_name] / 100
output_array = xr.full_like(data_array, NaN)
# output_array = fill(NaN, width_raster, height_raster)
# Now we iterate over the buffered windows
for i in 0:(x_steps-1)
    for j in 0:(y_steps-1)
        x_start = i * step_size
        y_start = j * step_size
        # Extract the buffered window from the raw dataset
        buffered_window = data_array.isel(
            x=pyslice(x_start, x_start + total_window_size),
            y=pyslice(y_start, y_start + total_window_size)
            )
        hab_qual = xr_dataarray_to_array(buffered_window)
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
            # This is not working
            # output_array.isel(x = pyslice(x_start + buffer_size, x_start + buffer_size + window_size),
            #                 y = pyslice(y_start + buffer_size, y_start + buffer_size + window_size)) .= sensitivities
            range = buffer_size:(buffer_size+window_size)
            @show x_start, y_start
            output_array.values[0, pyslice(x_start + buffer_size, x_start + buffer_size + window_size), pyslice(y_start + buffer_size, y_start + buffer_size + window_size)] = sensitivities[buffer_size+1:(buffer_size+window_size), buffer_size+1:(buffer_size+window_size)]
        end
    end
end

using Plots
heatmap(pyconvert(Array, output_array.values)[1, :, :])

#=
TODO:
- We need to find a way to incorporate back `sensitivities` into the xarray
- The trick using a Julia array to store output works, but something is wrong with the coordinates - check plot
=#

fig, ax = plt.subplots()
output_array.plot(ax=ax)
fig


# D = 1
hab_qual = xr_dataarray_to_array(dataset[sp_name])

# scaling
hab_qual = hab_qual ./ 100.

g = GridGraph(hab_qual; vertex_activities = hab_qual .> cut_off)

# calculating distance matrix:
euclidean_distance = calculate_euclidean_distance(g, res)

# calculate proximity
K = exp.(-euclidean_distance / D)

q = [hab_qual[ij...] for ij in id_to_grid_coordinate_list(g)]

# K is ecologicla proximity, q is a list of suitability values for active vertices





calculate_functional_habitat(q, K)

sensitivities_vec = gradient(q -> calculate_functional_habitat(q, K), q)[1]

# plotting sensitivity
sensitivities = fill(NaN, height(g), width(g))
[sensitivities[ij...] = sensitivities_vec[v] for (v, ij) in enumerate(id_to_grid_coordinate_list(g))]

xr_sensitivities = dataset[sp_name].copy()
xr_sensitivities.values = reshape(sensitivities, (1, size(sensitivities)...))



dataset[sp_name*("dfunc_dqual")] = xr_sensitivities
file_name = splitext(basename(@__FILE__))[1] * ".nc"
output_path = mkdir(joinpath(@__DIR__, "data"))
dataset.to_netcdf(joinpath(output_path, file_name), engine="netcdf4")


dataset[sp_name*("dfunc_dqual")].plot(ax=ax)