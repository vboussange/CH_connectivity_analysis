#=
Run zonation algorithm on each guild separately
=#

cd(@__DIR__)
using PythonCall
include("../src/landscape.jl") # orders matters!
using ConScape
import ConScape:_targetidx_and_nodes
using SparseArrays, DataFrames, CSV
using ProgressBars

dataset_path = joinpath(@__DIR__, "../../data/GUILDES_EU_buffer_dist=50km_resampling_2.nc")
θ = 0.1
α = 75 # movement capability
guild_idx = 1
nb_cells_to_discard = 1000

dataset = load_xr_dataset(dataset_path)
guild_names = pyconvert(Vector, dataset.data_vars)
guilde_arrays = xr_dataset_to_array(dataset)

hab_qual = guilde_arrays[guild_idx, :, :]
# scaling
hab_qual = hab_qual ./ maximum(hab_qual[.!isnan.(hab_qual)])

adjacency_matrix = ConScape.graph_matrix_from_raster(hab_qual)
original_grid = ConScape.Grid(size(hab_qual)...,
                        affinities=adjacency_matrix,
                        source_qualities=hab_qual,
                        target_qualities=ConScape.sparse(hab_qual),
                        costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))
it = 0
cell_importance = fill(CartesianIndex{2}(), length(original_grid.id_to_grid_coordinate_list))
total_func_hab = sum(filter(!isnan, hab_qual))

# nb_iterations = 

while total_func_hab > 0
    adjacency_matrix .= ConScape.graph_matrix_from_raster(hab_qual)

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

    nb_cells_to_discard = min(nb_cells_to_discard, length(g.id_to_grid_coordinate_list))
    sorted_idx = sortperm(hab_qual[g.id_to_grid_coordinate_list])[1:nb_cells_to_discard]
    cells_to_discard = g.id_to_grid_coordinate_list[sorted_idx[1:nb_cells_to_discard]]
    cell_importance[it * nb_cells_to_discard + 1:(it +1) * nb_cells_to_discard] = cells_to_discard
    hab_qual[cells_to_discard] .= 0
    it+=1
    total_func_hab = sum(filter(!isnan, hab_qual))
end

# mapping importance
importance_raster = fill(NaN, original_grid.nrows, original_grid.ncols)
for (ij, x) in zip(cell_importance, range(0, 1, length=length(cell_importance)))
    importance_raster[ij] = x
end
Plots.heatmap(importance_raster)
Plots.heatmap(hab_qual)