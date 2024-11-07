using CSV
using DataFrames
using Printf

const TRAITS_CH_PATH = joinpath(@__DIR__, "../../data/TraitsCH/dispersal_focus/S_MEAN_dispersal_all.txt")

struct TraitsCH
    data::DataFrame
end

function TraitsCH()
    df = CSV.read(TRAITS_CH_PATH, DataFrame; delim=' ')
    # Dummy placeholder, to be changed
    df[!, :habitat] .= "Aqu"
    return TraitsCH(df)
end

function get_key(traits::TraitsCH, species_name::String, key::Symbol)
    species_row = traits.data[traits.data.Species .== species_name, :]
    if !isempty(species_row)
        return species_row[1, key]
    else
        throw(ArgumentError("Species '$species_name' not found in the dataset."))
    end
end

function get_habitat(traits::TraitsCH, species_name::String)
    return get_key(traits, species_name, :habitat)
end

function get_D(traits::TraitsCH, species_name::String)
    # Check for realistic values
    return get_key(traits, species_name, :Dispersal_km)
end