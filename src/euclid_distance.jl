function calculate_euclidean_distance(g::Grid, res)
    coordinate_list = active_vertices_coordinate(g)
    euclidean_distance = [hypot(xy_i[1] - xy_j[1], xy_i[2] - xy_j[2]) for xy_i in coordinate_list, xy_j in coordinate_list]
    return euclidean_distance * res
end


function calculate_euclidean_distance_gpu(g::Grid, res)
    coordinate_list = active_vertices_coordinate(g)
    XY = CuArray{Float32}(stack(coordinate_list))
    Xmat = XY[1,:] .- XY[1,:]'
    Ymat = XY[2,:] .- XY[2,:]'
    euclidean_distance = hypot.(Xmat, Ymat)
    return euclidean_distance * res
end