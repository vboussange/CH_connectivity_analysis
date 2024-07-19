#=
Run zonation algorithm on each guild separately
=#

cd(@__DIR__)
using PythonCall
include("../../src/landscape.jl") # orders matters!
using ConScape
import ConScape:_targetidx_and_nodes
using SparseArrays, DataFrames, CSV
using ProgressBars
using DataFrames
using JLD2
using Printf
using PythonPlot

dataset_path = joinpath(@__DIR__, "../../data/GUILDES_EU_buffer_dist=50km_resampling_4.nc")
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
g = ConScape.Grid(size(hab_qual)...,
                        affinities=adjacency_matrix,
                        source_qualities=hab_qual,
                        target_qualities=ConScape.sparse(hab_qual),
                        costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))
coarse_target_qualities = ConScape.coarse_graining(g, 4)
g = ConScape.Grid(size(hab_qual)...,
                affinities=adjacency_matrix,
                source_qualities=hab_qual,
                target_qualities=coarse_target_qualities,
                costs=ConScape.mapnz(x -> -log(x), adjacency_matrix))

df = DataFrame(:α => [], :θ => [], :func => [], :bet => [])
for θ in exp.(range(log(0.01), log(2.5), length=4))
    @time h = ConScape.GridRSP(g, θ=θ)
    for α in exp.(range(log(1), log(200), length=4))
        # calculating functional habitat
        @time func = ConScape.connected_habitat(h, 
                                                connectivity_function = ConScape.expected_cost, 
                                                distance_transformation=x -> exp(-x/α));

        @time bet = ConScape.betweenness_kweighted(h,
                                                    distance_transformation=x -> exp(-x/α));
    push!(df, (α, θ, func, bet))
    end
end

jldsave("exploration_alpha_theta_guild_$(guild_idx).jld2"; df)

df = load("exploration_alpha_theta_guild_$(guild_idx).jld2", "df")

dfg = groupby(df, :α)
fig, axs = plt.subplots(4, 2, figsize= (5, 10))

for (i, df) in enumerate(dfg)
    sort!(df, :θ)
    func, bet = df[2, [:func, :bet]]
    α = df.α[1]
    # axs[i-1, 0].imshow(hab_qual)   
    # axs[0].set_title("Habitat quality")                 
    # fig

    axs[i-1, 0].imshow(func)
    axs[i-1, 1].imshow(bet)   

    if i == 1
        axs[i-1, 0].set_title(@sprintf("Functional habitat\nα = %.2f", α))    
        axs[i-1, 1].set_title("Betweenness")                 
    else             
        axs[i-1, 0].set_title(@sprintf("α = %.2f", α))    
    end

    # for ax in axs.flatten()
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # end

end
display(fig)
fig.savefig("exploration_alpha_guild_$(guild_idx).png", dpi=300)

dfg = groupby(df, :θ)
fig, axs = plt.subplots(4, 2, figsize= (5, 10))

for (i, df) in enumerate(dfg)
    sort!(df, :α)
    func, bet = df[3, [:func, :bet]]
    θ = df.θ[1]
    # axs[i-1, 0].imshow(hab_qual)   
    # axs[0].set_title("Habitat quality")                 
    # fig

    axs[i-1, 0].imshow(func)
    axs[i-1, 1].imshow(bet)   

    if i == 1
        axs[i-1, 0].set_title(@sprintf("Functional habitat\nθ = %.2f", θ))    
        axs[i-1, 1].set_title("Betweenness\n")                 
    else             
        axs[i-1, 0].set_title(@sprintf("θ = %.2f", θ))    
    end

    # for ax in axs.flatten()
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # end

end
display(fig)
fig.savefig("exploration_theta_guild_$(guild_idx).png", dpi=300)