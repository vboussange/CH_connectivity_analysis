# This file runs an omniscape test
cd(@__DIR__)
import Plots
using Omniscape
using Rasters
using ArchGDAL
# currently, does not work
# ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
# ENV["JULIA_PYTHONCALL_EXE"] = "/Users/victorboussange/projects/connectivity/connectivity_analysis/code/python/.env/bin/python"  # optional
# using PythonCall
# sys = pyimport("sys")
# sys.path.append("./../../../python/src")
# TraitsCH = pyimport("TraitsCH").TraitsCH

species_name = "Squalius cephalus"
quality_raster = Raster("output/$species_name/quality.tif", replace_missing=true)
resistance_raster = Raster("output/$species_name/resistance.tif", replace_missing=true)
print("Size raster: $(size(quality_raster))")
print("Number of cells: $(length(filter(!ismissing, quality_raster)))")

Plots.plot(quality_raster)
Plots.plot(resistance_raster)

quality = Array{Union{Float64, Missing}}(quality_raster)
resistance = Array{Union{Float64, Missing}}(resistance_raster)

# TODO: to fix; we should be able to use the quality raster directly
resistance = abs.(resistance ) /  maximum(filter(!ismissing, abs.(resistance)))

# Create a subsample of the quality raster
# quality = Array{Union{Float64, Missing}}(quality[1:5:end, 1:5:end])
# resistance = abs.(resistance[1:5:end, 1:5:end] ) /  maximum(filter(!ismissing, abs.(resistance[1:5:end, 1:5:end] )))
# Plots.heatmap(quality)
# Plots.heatmap(resistance)


# TODO: to fix; we could save a json file with the parameters below
resolution = 100 
D_m = 4511.5
radius = ceil(Int, D_m / 100)
config = Dict{String, String}(
    "radius" => "$radius",
    "block_size" => "1",
    "project_name" => "",
    "calc_normalized_current" => "false",
    "calc_flow_potential" => "false",
    "parallelize" => "true",
    "solver" => "cg+amg",
)

currmap = run_omniscape(config,
                        resistance;
                        source_strength=quality)
Plots.heatmap(currmap)
# Plots.heatmap(norm_current)
current_raster = deepcopy(quality_raster)
current_raster .= currmap
Rasters.write("output/$species_name/currmap_block_size_$(config["block_size"]).tif", current_raster, force=true)


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