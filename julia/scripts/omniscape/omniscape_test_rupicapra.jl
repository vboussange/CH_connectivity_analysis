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

quality_raster = Raster("inputs/rupicapra/source.tif", replace_missing=true) / 100

Plots.plot(quality_raster)
quality_raster = Array{Union{Float64, Missing}}(quality_raster)
Plots.heatmap(quality_raster)

resistance_raster = - log.(quality_raster) .+ 0.1
Plots.heatmap(resistance_raster)

# Specify the configuration options
config = Dict{String, String}(
    "radius" => "100",
    "block_size" => "1",
    "project_name" => "",
    "calc_normalized_current" => "true",
    "calc_flow_potential" => "true",
    "parallelize" => "true",
    "solver" => "cg+amg"
)

currmap, flow_pot, norm_current = run_omniscape(config,
                                                resistance_raster,
                                                source_strength=quality_raster)
Plots.heatmap(currmap)
Plots.heatmap(norm_current)