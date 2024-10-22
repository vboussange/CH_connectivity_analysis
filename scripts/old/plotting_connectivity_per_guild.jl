
cd(@__DIR__)
using PythonCall
include("../src/landscape.jl") # orders matters!
using ConScape
import ConScape: _targetidx_and_nodes
using SparseArrays, DataFrames, CSV
using ProgressBars

dataset_path = joinpath(@__DIR__, "../../data/GUILDES_EU_buffer_dist=50km_resampling_4.nc")
output_path = joinpath(@__DIR__, "../../results/$(today())/connectivity_plot/" )
mkpath(output_path)

θ = 0.1
α = 75 # movement capability

dataset = load_xr_dataset(dataset_path)
guild_names = pyconvert(Vector, dataset.data_vars)
guilde_arrays = xr_dataset_to_array(dataset)

for (guild_idx, guild) in enumerate(guild_names)
    hab_qual = guilde_arrays[guild_idx, end:-1:1, :] # dirty fix to revert order
    hab_qual = hab_qual ./ maximum(hab_qual[.!isnan.(hab_qual)])
    adjacency_matrix = ConScape.graph_matrix_from_raster(hab_qual)

    g = ConScape.Grid(size(hab_qual)...,
    affinities=adjacency_matrix,
    source_qualities=hab_qual,
    target_qualities=ConScape.sparse(hab_qual),
    costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))


    coarse_target_qualities = ConScape.coarse_graining(g, 5)
    g = ConScape.Grid(size(hab_qual)...,
    affinities=adjacency_matrix,
    source_qualities=hab_qual,
    target_qualities=coarse_target_qualities,
    costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))

    @time h = ConScape.GridRSP(g, θ=θ)

    # calculating functional habitat
    @time connectity_map = ConScape.betweenness_kweighted(h, 
                            distance_transformation=x -> exp(-x/α));
    p = heatmap(connectity_map, 
                axis = false,# Removes the axes
                framestyle = :none,  # Removes the frame
                title = "",          # No title
                xlabel = "",         # No x-axis label
                ylabel = "",         # No y-axis label
                xticks = false,      # No x-axis ticks
                yticks = false,      # No y-axis ticks
                background_color = :transparent, # Set background to transparent,
                dpi=300,
                colorbar=false
                )
    savefig(p, joinpath(output_path, "$guild.png"))
end
