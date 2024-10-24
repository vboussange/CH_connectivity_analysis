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

dataset_path = joinpath(@__DIR__, "../../../data/GUILDS_EU_SP/GUILDS_EU_SP_Zug_resampling_1.nc")

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
    
dataset = load_xr_dataset(dataset_path)
trait_dataset = TraitsCH()

sp_name = "Salmo trutta"

D = pyconvert(Float64, trait_dataset.get_D(sp_name))
hab_qual = xr_dataarray_to_array(dataset[sp_name])

# scaling
hab_qual = hab_qual ./ 100.
cut_off = 0.1

g = GridGraph(hab_qual; vertex_activities = hab_qual .> cut_off)

# calculating distance matrix:
res = pyconvert(Float64, Preprocessing.calculate_resolution(dataset)[1]) / 1000 #km
euclidean_distance = calculate_euclidean_distance(g, res)

# calculate proximity
K = exp.(-euclidean_distance / D)

q = [hab_qual[ij...] for ij in id_to_grid_coordinate_list(g)]

# K is ecologicla proximity, q is a list of suitability values for active vertices
function calculate_functional_habitat(q, K)
    return sum(q .* (K * q))
end

calculate_functional_habitat(q, K)

sensitivities_vec = gradient(q -> calculate_functional_habitat(q, K), q)[1]

# plotting sensitivity
sensitivities = fill(NaN, height(g), width(g))
[sensitivities[ij...] = sensitivities_vec[v] for (v, ij) in enumerate(id_to_grid_coordinate_list(g))]

xr_sensitivities = dataset[sp_name].copy()
xr_sensitivities.values = reshape(sensitivities, (1, size(sensitivities)...))

fig, ax = plt.subplots()
xr_sensitivities.plot(ax=ax)
fig