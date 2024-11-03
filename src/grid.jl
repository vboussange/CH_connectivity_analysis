using SparseArrays, LinearAlgebra
using Graphs, SimpleWeightedGraphs, ProgressMeter

struct Grid{R<:Real, A <:SparseMatrixCSC{R, Int}, CM, VA <: AbstractMatrix{Bool}, SQ <: Matrix{R}, TQ <: AbstractMatrix{R}}
    affinities::A
    cost_matrix::CM
    vertex_activities::VA
    source_qualities::SQ
    target_qualities::TQ
end


function Grid(habitat_suitability, affinity_matrix; C = x -> -log(x),  prune=true)
    # when `prune==true`, we activate only vertices belonging to the largest component
    if prune
        active_vertices = largest_subgraph_vertices(affinity_matrix)
        activity_matrix = fill(false, size(habitat_suitability))
        activity_matrix[active_vertices] .= true
    else
        activity_matrix = fill(true, size(habitat_suitability))
    end

    grid = Grid(affinity_matrix,
                C,
                activity_matrix,
                Matrix(habitat_suitability),
                Matrix(habitat_suitability))
    return grid
end


## Attribute access
vertex_activities(g::Grid) = g.vertex_activities
affinities(g::Grid) = g.affinities
cost_matrix(g::Grid) = g.cost_matrix
source_qualities(g::Grid) = g.source_qualities
target_qualities(g::Grid) = g.target_qualities
Graphs.nv(g::Grid) = height(g) * width(g)
Graphs.vertices(g::Grid) = 1:nv(g)
Graphs.has_vertex(g::Grid, v) = 1 <= v <= nv(g)

## Size
"""
    height(g)

Compute the height of the grid (number of rows).
"""
height(g::Grid) = size(source_qualities(g), 1)


"""
    width(g)

Compute the width of the grid (number of columns).
"""
width(g::Grid) = size(source_qualities(g), 2)

Base.size(g::Grid) = (width(g), height(g))

## Indexing

"""
    coord_to_index(g, i, j)

Convert a grid coordinate tuple `(i,j)` into the index `v` of the associated vertex.
"""
function coord_to_index(g::Grid, i, j)
    h, w = height(g), width(g)
    if (1 <= i <= h) && (1 <= j <= w)
        v = (j - 1) * h + (i - 1) + 1  # enumerate column by column
        return v
    else
        return 0
    end
end

"""
    index_to_coord(g, v)

Convert a vertex index `v` into the tuple `(i,j)` of associated grid coordinates.
"""
function index_to_coord(g::Grid, v)
    if has_vertex(g, v)
        h = height(g)
        j = (v - 1) ÷ h + 1
        i = (v - 1) - h * (j - 1) + 1
        return (i, j)
    else
        return (0, 0)
    end
end

## Activity

"""
    vertex_active(g, v)

Check if vertex `v` is active.
"""
vertex_active(g::Grid, v) = vertex_activities(g)[v]
vertex_active_coord(g::Grid, i, j) = vertex_activities(g)[i, j]
nb_active(g::Grid) = sum(vertex_activities(g))
all_active(g::Grid) = nb_active(g) == length(vertex_activities(g))
list_active_vertices(grid::Grid) = [v for v in vertices(grid) if vertex_active(grid, v)]
active_vertices_coordinate(g::Grid) = [index_to_coord(g, v) for v in list_active_vertices(g)]

function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    print(io, "Grid of size ", height(g), "x", width(g))
end

function Base.show(io::IO, ::MIME"text/html", g::Grid)
    t = string(summary(g), " of size ", g.nrows, "x", g.ncols)
    write(io, "<h4>$t</h4>")
    write(io, "<table><tr><td>Affinities</br>")
    show(io, MIME"text/html"(), plot_outdegrees(g))
    write(io, "</td></tr></table>")
    if g.source_qualities === g.target_qualities
        write(io, "<table><tr><td>Qualities</td></tr></table>")
        show(io, MIME"text/html"(), heatmap(g.source_qualities, yflip=true))
    else
        write(io, "<table><tr><td>Source qualities")
        show(io, MIME"text/html"(), heatmap(g.source_qualities, yflip=true))
        write(io, "</td><td>Target qualities")
        show(io, MIME"text/html"(), heatmap(Matrix(g.target_qualities), yflip=true))
        write(io, "</td></tr></table>")
    end
end

function _targetidx_and_nodes(grid::Grid)
    # returns all active vertices which target quality is higher than 0
    vertices = [v for v in list_active_vertices(grid) if grid.target_qualities[v] > 0.]
    indices = [index_to_coord(grid, v) for v in vertices]
    return indices, vertices
end


"""
    is_strongly_connected(g::Grid)::Bool

Test if graph defined by Grid is fully connected.

# Examples

```jldoctests
julia> affinities = [1/4 0 1/4 1/4
                     1/4 0 1/4 1/4
                     1/4 0 1/4 1/4
                     1/4 0 1/4 1/4];

julia> grid = ConScape.Grid(size(affinities)..., affinities=ConScape.graph_matrix_from_raster(affinities), prune=false)
ConScape.Grid of size 4x4

julia> ConScape.is_strongly_connected(grid)
false
```
"""
Graphs.is_strongly_connected(g::Grid) = is_strongly_connected(SimpleWeightedDiGraph(affinities(g)))

"""
    largest_subgraph(g::Grid)::Grid

Extract the largest fully connected subgraph of the `Grid`. The returned `Grid`
will have the same size as the input `Grid` but only nodes associated with the
largest subgraph of the affinities will be active.
"""
function largest_subgraph_vertices(affinity_matrix)
    # Convert affinity matrix to graph
    graph = SimpleWeightedDiGraph(affinity_matrix, permute=false)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    @debug "cost graph contains $(length(scc)) strongly connected subgraphs"

    # Find the largest subgraph
    i = argmax(length.(scc))

    # extract node list and sort it
    scci = sort(scc[i])

    ndiffnodes = size(affinity_matrix, 1) - length(scci)
    if ndiffnodes > 0
        @debug "removing $ndiffnodes nodes from affinity and cost graphs"
    end
    # largest_subgraph_affinity_matrix = spzeros(size(affinity_matrix))
    # Extract the adjacency matrix of the largest subgraph
    # largest_subgraph_affinity_matrix[scci, scci] = affinity_matrix[scci, scci]
    return scci
end

"""
    least_cost_distance(g::Grid)::Matrix{Float64}

Compute the least cost distance from all the cells in the grid to all target cells.

# Examples
```jldoctests
julia> affinities = [1/4 0 1/2 1/4
                     1/4 0 1/2 1/4
                     1/4 0 1/2 1/4
                     1/4 0 1/2 1/4];

julia> grid = ConScape.Grid(size(affinities)..., affinities=ConScape.graph_matrix_from_raster(affinities))
[ Info: cost graph contains 6 strongly connected subgraphs
[ Info: removing 8 nodes from affinity and cost graphs
ConScape.Grid of size 4x4

julia> ConScape.least_cost_distance(grid)
8×8 Matrix{Float64}:
 0.0       0.693147  1.38629   2.07944   0.693147  1.03972   1.73287   2.42602
 0.693147  0.0       0.693147  1.38629   1.03972   0.693147  1.03972   1.73287
 1.38629   0.693147  0.0       0.693147  1.73287   1.03972   0.693147  1.03972
 2.07944   1.38629   0.693147  0.0       2.42602   1.73287   1.03972   0.693147
 1.38629   1.73287   2.42602   3.11916   0.0       1.38629   2.77259   3.46574
 1.73287   1.38629   1.73287   2.42602   1.38629   0.0       1.38629   2.77259
 2.42602   1.73287   1.38629   1.73287   2.77259   1.38629   0.0       1.38629
 3.11916   2.42602   1.73287   1.38629   3.46574   2.77259   1.38629   0.0
```
"""
function least_cost_distance(g::Grid; θ::Nothing=nothing, approx::Bool=false)
    # FIXME! This should be multithreaded. However, ProgressLogging currently
    # does not support multithreading
    if approx
        throw(ArgumentError("no approximate algorithm is available for this distance function"))
    end
    targets = ConScape._targetidx_and_nodes(g)[1]
    vec_of_vecs = @showprogress [_least_cost_distance(g, target) for target in targets]

    return reduce(hcat, vec_of_vecs)
end

function _least_cost_distance(g::Grid, target::CartesianIndex{2})
    graph = SimpleWeightedDiGraph(g.costmatrix)
    targetnode = findfirst(isequal(target), g.id_to_grid_coordinate_list)
    distvec = dijkstra_shortest_paths(graph, targetnode).dists
    return distvec
end

function _vec_to_grid(g::Grid, vec::Vector)
    grid = fill(Inf, g.nrows, g.ncols)
    for (i, c) in enumerate(g.id_to_grid_coordinate_list)
        grid[c] = vec[i]
    end
    return grid
end

"""
    sum_neighborhood(g::Grid, rc::Tuple{Int,Int}, npix::Integer)::Float64

A helper-function, used by coarse_graining, that computes the sum of pixels within a npix neighborhood around the target rc.
"""
function sum_neighborhood(g, rc, npix)
    getrows = (rc[1] - floor(Int, npix/2)):(rc[1] + (ceil(Int, npix/2) - 1))
    getcols = (rc[2] - floor(Int, npix/2)):(rc[2] + (ceil(Int, npix/2) - 1))
    # pixels outside of the landscape are encoded with NaNs but we don't want
    # the NaNs to propagate to the coarse grained values
    return sum(t -> isnan(t) ? 0.0 : t, g.target_qualities[getrows, getcols])
end



"""
    coarse_graining(g::Grid, npix::Integer)::Array

Creates a sparse matrix of target qualities for the landmarks based on merging npix pixels into the center pixel.
"""
function coarse_graining(g, npix)
    getrows = (floor(Int, npix/2)+1):npix:(g.nrows-ceil(Int, npix/2)+1)
    getcols = (floor(Int, npix/2)+1):npix:(g.ncols-ceil(Int, npix/2)+1)
    coarse_target_rc = Base.product(getrows, getcols)
    coarse_target_ids = vec(
        [
            findfirst(
                isequal(CartesianIndex(ij)),
                g.id_to_grid_coordinate_list
            ) for ij in coarse_target_rc
        ]
    )
    coarse_target_rc = [ij for ij in coarse_target_rc if !ismissing(ij)]
    filter!(!ismissing, coarse_target_ids)
    V = [sum_neighborhood(g, ij, npix) for ij in coarse_target_rc]
    I = first.(coarse_target_rc)
    J = last.(coarse_target_rc)
    target_mat = sparse(I, J, V, g.nrows, g.ncols)
    target_mat = dropzeros(target_mat)

    return target_mat
end