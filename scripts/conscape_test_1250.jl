# Here we import a sample guild habitat suitability, and we 
# calculate adjacency_matrix, which we save to be imported e.g. in Python
# we then attempt to convert it into a ConScape.Grid, although we are not sure about how to define the costs (see ref in ConScape.jl paper)

using Rasters
import ArchGDAL
cd(@__DIR__)
using ConScape
using SparseArrays, DataFrames, CSV
using Plots


LEGEND = Dict(14 => "prairies et pâturages secs; prairies grasses riches en espèces",
    8 => "forêts alluviales",
    2 => "cours d'eau dynamiques et leurs rives")

RESOLUTION = "1250m"
dir = "img/"
θ = 0.5

for k in keys(LEGEND)
    hab_qual = Raster("../data/hsv_guilde_$(k)_$(RESOLUTION).tif")
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

    ConScape.heatmap(g.source_qualities, yflip=true,
        title="Map of habitat quality, 
        $(LEGEND[k])", titlefontsize=8,
        background_color = :transparent,
        color=cgrad([:white, :green]))
    savefig(dir * "habitat_quality_$(k)_$(RESOLUTION).png")

    ConScape.plot_outdegrees(g, title="Map of permeability to movement", color=cgrad(:acton), background_color = :transparent)

    # calculation of RSP paths
    @time h = ConScape.GridRSP(g, θ=θ)


    # Computing the distance from all pixels in the landscape to a given target pixel.
    # nnodes = g.nrows * g.ncols
    # tmp = zeros(nnodes)
    # tmp[10] = 1
    # ConScape.plot_values(g, tmp, title="One target pixel t") #errors

    func = ConScape.connected_habitat(h,
        connectivity_function=ConScape.expected_cost,
        distance_transformation=x -> exp(-x / 75))


    ConScape.heatmap(Array(func), yflip=true, title="Functional habitat, $(LEGEND[k])", titlefontsize=8)
    savefig(dir * "functional_habitat_$(k)_$(RESOLUTION)_theta_$(θ).png")

    # landscape level connectivity
    sum(func)
    sum(filter(!isnan, func))

    # compare this value to the amount of ‘unconnected’ habitat, we see that there is some loss of habitat due to movement constraints
    100 * (1 - sqrt(sum(filter(x -> !isnan(x), func))) /
               sum(filter(x -> !isnan(x), g.source_qualities)))

    # Computation of movement flow
    kbetw = ConScape.betweenness_kweighted(h,
        distance_transformation=x -> exp(-x / 75))
    ConScape.heatmap(kbetw, yflip=true, title="Betweenness, $(LEGEND[k])", titlefontsize=8, background_color = :transparent)
    savefig(dir * "flow_movement_$(k)_$(RESOLUTION)_theta_$(θ).png")
end