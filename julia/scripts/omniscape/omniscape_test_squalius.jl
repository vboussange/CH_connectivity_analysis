# This file runs an omniscape test
cd(@__DIR__)
using GridGraphs, Graphs
using SparseArrays, DataFrames, CSV
using ProgressMeter
using DataFrames
using JLD2
using Printf
using PythonPlot
import Plots
using Omniscape
using Rasters
using ArchGDAL

quality_raster = Raster("inputs/squalius/quality_small.tif", replace_missing=true) / 100
print("Size raster: $(size(quality_raster))")
print("Number of cells: $(length(filter(!ismissing, quality_raster)))")

Plots.plot(quality_raster)
quality = Array{Union{Float64, Missing}}(quality_raster)
Plots.heatmap(quality)

resistance = - log.(quality) .+ 0.1
Plots.heatmap(resistance)

# Block_size = 1
config = Dict{String, String}(
    "radius" => "200",
    "block_size" => "1",
    "project_name" => "",
    "calc_normalized_current" => "true",
    "calc_flow_potential" => "true",
    "parallelize" => "true",
    "solver" => "cg+amg",
)

currmap, flow_pot, norm_current = run_omniscape(config,
                                                resistance,
                                                source_strength=quality)
Plots.heatmap(currmap)
Plots.heatmap(norm_current)
current_raster = deepcopy(quality_raster)
current_raster .= currmap
Rasters.write("outputs/squalius/currmap_block_size_$(config["block_size"]).tif", current_raster, force=true)


# Block_size = 9
config = Dict{String, String}(
    "radius" => "200",
    "block_size" => "9",
    "project_name" => "",
    "calc_normalized_current" => "true",
    "calc_flow_potential" => "true",
    "parallelize" => "true",
    "solver" => "cg+amg",
)

currmap, flow_pot, norm_current = run_omniscape(config,
                                                resistance,
                                                source_strength=quality)
Plots.heatmap(currmap)
Plots.heatmap(norm_current)
current_raster = deepcopy(quality_raster)
current_raster .= currmap
Rasters.write("outputs/squalius/currmap_block_size_$(config["block_size"]).tif", current_raster, force=true)


# Block_size = 21
config = Dict{String, String}(
    "radius" => "200",
    "block_size" => "21",
    "project_name" => "",
    "calc_normalized_current" => "true",
    "calc_flow_potential" => "true",
    "parallelize" => "true",
    "solver" => "cg+amg",
)

currmap, flow_pot, norm_current = run_omniscape(config,
                                                resistance,
                                                source_strength=quality)
Plots.heatmap(currmap)
Plots.heatmap(norm_current)
current_raster = deepcopy(quality_raster)
current_raster .= currmap
Rasters.write("outputs/squalius/currmap_block_size_$(config["block_size"]).tif", current_raster, force=true)