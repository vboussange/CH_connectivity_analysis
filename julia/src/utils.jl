function resolution(ras::Raster)
    lon = lookup(ras, X) 
    return lon[2] - lon[1]
end

function calculate_functional_habitat(q, K)
    return sum(q .* (K * q))
end