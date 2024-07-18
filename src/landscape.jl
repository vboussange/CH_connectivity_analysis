using PythonCall
xr = pyimport("xarray")
rioxarray = pyimport("rioxarray")
plt = pyimport("matplotlib.pyplot")
gpd = pyimport("geopandas")
using ConScape: N8

const SWISS_BOUNDARY_FILE = joinpath(@__DIR__(), "../../data/swiss_boundaries/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp")

function load_xr_raster(path)
    da = rioxarray.open_rasterio(path, mask_and_scale=true)
    # we load slightly more than the extent to be able to correctly interpolate
    # cda = da.sel(band=1)
    # temp_raster = pyconvert(Array{Float32}, cda.drop_vars("band").to_numpy()) # we keep `spatial_ref` var. as it contains crs data
    da.close()
    return da
end

function load_xr_dataset(path)
    da = rioxarray.open_dataset(path)
    # we load slightly more than the extent to be able to correctly interpolate
    # cda = da.sel(band=1)
    # temp_raster = pyconvert(Array{Float32}, cda.drop_vars("band").to_numpy()) # we keep `spatial_ref` var. as it contains crs data
    da.close()
    return da
end

"""

buffer_distance in meter
"""
function crop_raster(raster, buffer_distance)
    switzerland_boundary = gpd.read_file(SWISS_BOUNDARY_FILE)
    switzerland_boundary = switzerland_boundary[switzerland_boundary.ICC == Py("CH")]
    switzerland_buffer = switzerland_boundary.buffer(buffer_distance)
    if (switzerland_buffer.crs != raster.rio.crs) |> Bool
        raster = raster.rio.reproject(switzerland_buffer.crs)
    end
    buffered_gdf = gpd.GeoDataFrame(geometry=switzerland_buffer)
    masked_raster = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    return masked_raster
end

function graph_matrix_from_raster(
    R;
    neighbors::Tuple=N8,
)
    m, n = size(R)

    # Initialize the buffers of the SparseMatrixCSC
    is, js, vs = Int[], Int[], Float64[]

    for j in 1:n
        for i in 1:m
            # Base node
            for (ki, kj, l) in neighbors
                if !(1 <= i + ki <= m) || !(1 <= j + kj <= n)
                    # Continue when computing edge out of raster image
                    continue
                else
                    # Target node
                    rijk = R[i+ki, j+kj]
                    if iszero(rijk) || isnan(rijk)
                        # Don't include zero or NaN similaritiers
                        continue
                    end
                    push!(is, (j - 1) * m + i)
                    push!(js, (j - 1) * m + i + ki + kj * m)
                    push!(vs, rijk * l)
                end
            end
        end
    end
    return sparse(is, js, vs, m * n, m * n)
end

struct Landscape{M,AT,D,ID}
    nrows::M # raster nb rows
    ncols::M # raster nb cols
    A::AT # adjacency matrix
    dists::D # distance matrix
    id_to_grid_coordinate_list::ID
end

function Landscape(raster)
    nrows, ncols = size(raster)
    A = graph_matrix_from_raster(raster)
    id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))
    
    # pruning
    scci = largest_subgraph(SimpleWeightedDiGraph(A))
    g = SimpleWeightedDiGraph(A[scci, scci])
    id_to_grid_coordinate_list = id_to_grid_coordinate_list[scci]

    # calculating distance
    dists = floyd_warshall_shortest_paths(g, weights(g)).dists
    Landscape(nrows, ncols, A, dists, id_to_grid_coordinate_list)
end

function largest_subgraph(graph)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    @info "cost graph contains $(length(scc)) strongly connected subgraphs"

    # Find the largest subgraph
    i = argmax(length.(scc))

    # extract node list and sort it
    scci = sort(scc[i])
    A = adjacency_matrix(graph)

    ndiffnodes = size(A, 1) - length(scci)
    if ndiffnodes > 0
        @info "removing $ndiffnodes nodes from affinity and cost graphs"
    end

    return scci
end

function get_raster_values(values, l::Landscape)
    canvas = fill(NaN, l.nrows, l.ncols)
    for (i, v) in enumerate(values)
        canvas[l.id_to_grid_coordinate_list[i]] = v
    end
    return canvas
end
