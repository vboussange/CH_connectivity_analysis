# Here we import a sample guild habitat suitability, and we 
# calculate adjacency_matrix, which we save to be imported e.g. in Python
# we then attempt to convert it into a ConScape.Grid, although we are not sure about how to define the costs (see ref in ConScape.jl paper)
cd(@__DIR__)

using PythonCall
include("../src/landscape.jl") # orders matters!
using ConScape
using SparseArrays, DataFrames, CSV

RESOLUTION = "1250m"
dir = "img/"
θ = 0.5
k = 14

# reading raster and cropping with buffer
buffer_distance = 100000.
raster = load_xr_dataset("../../data/GUILDES_EU/GUILDE_1.tif")
raster = crop_raster(raster, buffer_distance)
fig, ax = plt.subplots()
raster.plot(ax=ax)
fig

hab_qual = replace_missing(hab_qual, NaN)
hab_qual = Matrix(transpose(Array(hab_qual)[:, :, 1]))

# scaling
hab_qual = hab_qual ./ maximum(hab_qual[.!isnan.(hab_qual)])

adjacency_matrix = ConScape.graph_matrix_from_raster(hab_qual)

I, J, V = findnz(adjacency_matrix)
df = DataFrame([:I => I, :J => J, :V => V])
CSV.write("/tmp/spmatrix.csv", df)

g = ConScape.Grid(size(hab_qual)...,
    affinities=adjacency_matrix,
    source_qualities=hab_qual,
    target_qualities=ConScape.sparse(hab_qual),
    costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))

coarse_target_qualities = ConScape.coarse_graining(g, 20)
g = ConScape.Grid(size(hab_qual)...,
    affinities=adjacency_matrix,
    source_qualities=hab_qual,
    target_qualities=coarse_target_qualities,
    costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))

@time h = ConScape.GridRSP(g, θ=θ)

kbetw = ConScape.betweenness_kweighted(h,
    distance_transformation=x -> exp(-x / 75))
ConScape.heatmap(log.(kbetw), yflip=true, title="Betweenness, $(LEGEND[k])", titlefontsize=8, background_color = :transparent)
savefig(dir * "flow_movement_$(k)_$(RESOLUTION)_theta_$(θ).png")
