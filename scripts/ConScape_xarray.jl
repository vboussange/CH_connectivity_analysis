# Here we import a sample guild habitat suitability, and we 
# calculate adjacency_matrix, which we save to be imported e.g. in Python
# we then attempt to convert it into a ConScape.Grid, although we are not sure about how to define the costs (see ref in ConScape.jl paper)
cd(@__DIR__)

using PythonCall
include("../src/landscape.jl") # orders matters!
using ConScape
using SparseArrays, DataFrames, CSV

dataset_path = joinpath(@__DIR__, "../../data/GUILDES_EU_buffer_dist=50km_resampling_2.nc")
θ = 0.1
guild_idx = 1
α = 75 # movement_capability

dataset = load_xr_dataset(dataset_path)
guild_names = pyconvert(Vector, dataset.data_vars)
guilde_arrays = xr_dataset_to_array(dataset)

hab_qual = guilde_arrays[guild_idx, :, :]
# scaling
hab_qual = hab_qual ./ maximum(hab_qual[.!isnan.(hab_qual)])

adjacency_matrix = ConScape.graph_matrix_from_raster(hab_qual)

g = ConScape.Grid(size(hab_qual)...,
                    affinities=adjacency_matrix,
                    source_qualities=hab_qual,
                    target_qualities=ConScape.sparse(hab_qual),
                    costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))

coarse_target_qualities = ConScape.coarse_graining(g, 50)
g = ConScape.Grid(size(hab_qual)...,
    affinities=adjacency_matrix,
    source_qualities=hab_qual,
    target_qualities=coarse_target_qualities,
    costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))

@time h = ConScape.GridRSP(g, θ=θ)


# calculating functional habitat
@time func = ConScape.connected_habitat(h, 
                    connectivity_function = ConScape.expected_cost,
                    distance_transformation=x -> exp(-x/α));
ConScape.heatmap(Array(func), yflip = true, title = "Functional habitat")

# landscape-level functionality https://www.sciencedirect.com/science/article/pii/S0169204607000709
sum(filter(!isnan, func))

#  amount of connected habitat https://www.sciencedirect.com/science/article/pii/S1470160X10001159
sqrt(sum(filter(!isnan, func)))

# amount of ‘unconnected’ habitat
100*(1-sqrt(sum(filter(!isnan, func)))/
            sum(filter(!isnan, g.source_qualities)))



kbetw = ConScape.betweenness_kweighted(h,
    distance_transformation=x -> exp(-x / 75))
ConScape.heatmap(log.(kbetw), yflip=true, title="Betweenness, $(guild_names[guild_idx])", titlefontsize=8, background_color = :transparent)
