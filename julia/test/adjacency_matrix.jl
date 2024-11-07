using ConScape

permeability_raster = ones(2,3)
permeability_raster[1,3] = 5
affinity_matrix = ConScape.graph_matrix_from_raster(permeability_raster, neighbors=ConScape.N4)

grid = ConScape.Grid(size(permeability_raster)...,
                        affinities=affinity_matrix,
                        source_qualities=Matrix(permeability_raster),
                        target_qualities=sparse(permeability_raster),
                        costs = affinity_matrix)

# to implement: make a test function

# function get_canvas(g::ConScape.Grid, values)
#     canvas = fill(NaN, g.nrows, g.ncols)
#     for (i,v) in enumerate(values)
#         canvas[g.id_to_grid_coordinate_list[i]] = v
#     end
#     return canvas
# end

# get_canvas(grid, affinity_matrix[:, 1])