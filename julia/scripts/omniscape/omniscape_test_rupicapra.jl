# This file runs an omniscape test
cd(@__DIR__)
import Plots
using Omniscape
using Rasters
using ArchGDAL
using JSON


species_name = "Rupicapra rupicapra"

path_folder = "output/$species_name"
json_file = joinpath(path_folder, "info.json")
config_data = JSON.parsefile(json_file)
quality_raster = Raster(joinpath(path_folder, "quality.tif"), replace_missing=true)
resistance_raster = Raster(joinpath(path_folder, "resistance.tif"), replace_missing=true)
println("Size raster: $(size(quality_raster))")
println("Number of cells: $(length(filter(!ismissing, quality_raster)))")

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



radius = ceil(Int, config_data["D_m"] / config_data["resolution"])
config = Dict{String, String}(
    "radius" => "$radius",
    "block_size" => "9",
    "project_name" => "",
    "calc_normalized_current" => "false",
    "calc_flow_potential" => "false",
    "parallelize" => "true",
    "solver" => "cg+amg",
    "precision" => "double"
)

currmap = run_omniscape(config,
                        resistance;
                        source_strength=quality)
Plots.heatmap(currmap)
# Plots.heatmap(norm_current)
current_raster = deepcopy(quality_raster)
current_raster .= currmap
Rasters.write("output/$species_name/currmap_block_size_$(config["block_size"]).tif", current_raster, force=true)
