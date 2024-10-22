using ConScape
using SparseArrays
using Rasters
using ArchGDAL
using ProgressBars
using DataFrames, Dates
using ThreadsX
using LinearAlgebra
using SimpleWeightedGraphs

findnearest(A::Vector{Float64}, t::Union{Float64,Int64}) = findmin(abs.(A .- t))[2]

# basic raster functions

function resolution(x::Union{Rasters.Raster,Rasters.RasterStack})
    x = [Rasters.lookup(x, X), Rasters.lookup(x, Y)]
    xres = collect(x[1])[2] - collect(x[1])[1]
    yres = collect(x[2])[1] - collect(x[2])[2]
    return ([xres, yres]')
end

function raster_bbox(x::Rasters.Raster)
    x = x.dims
    xres = collect(x[1])[2] - collect(x[1])[1]
    yres = collect(x[2])[1] - collect(x[2])[2]
    xmin = collect(x[1])[1]
    ymax = collect(x[2])[1] + yres
    xmax = last(collect(x[1])) + xres
    ymin = last(collect(x[2]))
    return ([xmin, xmax, ymin, ymax])
end

function extent(x::Rasters.Raster)
    x = x.dims
    xmin = first(collect(x[1]))
    ymax = first(collect(x[2])[1])
    xmax = last(collect(x[1]))
    ymin = last(collect(x[2]))
    return ([xmin, xmax, ymin, ymax])
end


# Silent grid creation

function silent_Grid(nrows::Integer,
                    ncols::Integer;
                    affinities=nothing,
                    qualities::Matrix=ones(nrows, ncols),
                    source_qualities::Matrix=qualities,
                    target_qualities::AbstractMatrix=qualities,
                    costs::Union{ConScape.Transformation,SparseMatrixCSC{Float64,Int}}=ConScape.MinusLog(),
                    prune=true)

    if affinities === nothing
        throw(ArgumentError("matrix of affinities must be supplied"))
    end

    if nrows * ncols != LinearAlgebra.checksquare(affinities)
        n = size(affinities, 1)
        throw(ArgumentError("grid size ($nrows, $ncols) is incompatible with size of affinity matrix ($n, $n)"))
    end

    _source_qualities = convert(Matrix{Float64}, source_qualities)
    _target_qualities = convert(AbstractMatrix{Float64}, target_qualities)

    # Prune
    # id_to_grid_coordinate_list = if prune
    #     nonzerocells = findall(!iszero, vec(sum(affinities, dims=1)))
    #     _affinities = affinities[nonzerocells, nonzerocells]
    #     vec(CartesianIndices((nrows, ncols)))[nonzerocells]
    # else
    #     _affinities = affinities
    #     vec(CartesianIndices((nrows, ncols)))
    # end
    id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))

    costfunction, costmatrix = if costs isa ConScape.Transformation
        costs, ConScape.mapnz(costs, affinities)
    else
        if nrows * ncols != LinearAlgebra.checksquare(costs)
            n = size(costs, 1)
            throw(ArgumentError("grid size ($nrows, $ncols) is incompatible with size of cost matrix ($n, $n)"))
        end
        nothing, costs
    end

    if any(t -> t < 0, nonzeros(costmatrix))
        throw(ArgumentError("The cost graph can have only non-negative edge weights. Perhaps you should change the cost function?"))
    end

    if SimpleWeightedGraphs.ne(SimpleWeightedGraphs.difference(SimpleWeightedGraphs.SimpleDiGraph(costmatrix), SimpleWeightedGraphs.SimpleDiGraph(affinities))) > 0
        throw(ArgumentError("cost graph contains edges not present in the affinity graph"))
    end

    g = ConScape.Grid(
        nrows,
        ncols,
        affinities,
        costfunction,
        costmatrix,
        id_to_grid_coordinate_list,
        _source_qualities,
        _target_qualities,
    )

    if prune
        return silent_largest_subgraph(g)
    else
        return g
    end
end

function silent_largest_subgraph(g::ConScape.Grid)
    # Convert cost matrix to graph
    graph = SimpleWeightedGraphs.SimpleWeightedDiGraph(g.costmatrix, permute=false)

    # Find the subgraphs
    scc = SimpleWeightedGraphs.strongly_connected_components(graph)

    #@info "cost graph contains $(length(scc)) strongly connected subgraphs"

    # Find the largest subgraph
    i = argmax(length.(scc))

    # extract node list and sort it
    scci = sort(scc[i])

    ndiffnodes = size(g.costmatrix, 1) - length(scci)
    if ndiffnodes > 0
        #    @info "removing $ndiffnodes nodes from affinity and cost graphs"
    end

    # Extract the adjacency matrix of the largest subgraph
    affinities = g.affinities[scci, scci]
    # affinities = convert(SparseMatrixCSC{Float64,Int}, graph[scci])

    return ConScape.Grid(
        g.nrows,
        g.ncols,
        affinities,
        g.costfunction,
        g.costfunction === nothing ? g.costmatrix[scci, scci] : ConScape.mapnz(g.costfunction, affinities),
        g.id_to_grid_coordinate_list[scci],
        g.source_qualities,
        g.target_qualities)
end


# Tiling functions 
"""
    make_tiles_table(rast, tile_size, overlap, cntr_size)

Create a `DataFrame` with tiles extracted from the `Raster` with a given tile_size and overlap in pixels.
"""

function make_tiles_table(rast::Union{Rasters.Raster,Rasters.RasterStack}, tile_size::Int64, overlap::Int64)
    #if tile_size < (radius*2 + cntr_size)
    #    throw("tile_size should allow at least one windowed computation")
    #end

    xdm = collect(Rasters.lookup(rast, X))
    ydm = collect(Rasters.lookup(rast, Y))

    xlen = ceil(size(xdm)[1] / tile_size)
    ylen = ceil(size(ydm)[1] / tile_size)

    tita = vec(collect(Iterators.product(collect(1:1:xlen), collect(1:1:ylen))))

    tita = DataFrames.DataFrame(ID=collect(1:size(tita)[1]), crds=tita)

    dms = Tuple[]
    dms_pdd = Tuple[]
    idx = Tuple[]
    idx_pdd = Tuple[]
    i = 1
    for i in (1:size(tita)[1])
        crd = tita[i, :crds]
        xs = [(first(crd) - 1) * tile_size + 1, first(crd) * tile_size, tile_size]
        if xs[2] > size(xdm)[1]
            xs[2] = size(xdm)[1]
            xs[3] = (xs[2] - xs[1]) + 1
        end
        xs = Int64.(xs)

        xs_pdd = copy(xs)
        if xs_pdd[1] != 1
            xs_pdd[1] += (-overlap)
        end
        if xs_pdd[2] != size(xdm)[1]
            xs_pdd[2] += overlap
        end
        if xs_pdd[2] > size(xdm)[1]
            xs_pdd[2] = size(xdm)[1]
        end
        xs_pdd[3] = (xs_pdd[2] - xs_pdd[1]) + 1
        xs_pdd = Int64.(xs_pdd)

        ys = [(last(crd) - 1) * tile_size + 1, last(crd) * tile_size, tile_size]
        if ys[2] > size(ydm)[1]
            ys[2] = size(ydm)[1]
            ys[3] = (ys[2] - ys[1]) + 1
        end
        ys = Int64.(ys)

        ys_pdd = copy(ys)
        if ys_pdd[1] != 1
            ys_pdd[1] += (-overlap)
        end
        if ys_pdd[2] != size(ydm)[1]
            ys_pdd[2] += overlap
        end
        if ys_pdd[2] > size(ydm)[1]
            ys_pdd[2] = size(ydm)[1]
        end
        ys_pdd[3] = (ys_pdd[2] - ys_pdd[1]) + 1
        ys_pdd = Int64.(ys_pdd)

        dms = push!(dms, (range(xdm[xs[1]], stop=xdm[xs[2]], length=xs[3]), range(ydm[ys[1]], stop=ydm[ys[2]], length=ys[3])))
        dms_pdd = push!(dms_pdd, (range(xdm[xs_pdd[1]], stop=xdm[xs_pdd[2]], length=xs_pdd[3]), range(ydm[ys_pdd[1]], stop=ydm[ys_pdd[2]], length=ys_pdd[3])))
        idx = push!(idx, ((xs[1]:xs[2]), (ys[1]:ys[2])))
        idx_pdd = push!(idx_pdd, ((xs_pdd[1]:xs_pdd[2]), (ys_pdd[1]:ys_pdd[2])))
    end

    tita[!, :dims] = dms
    tita[!, :padded_dims] = dms_pdd
    tita[!, :idx] = idx
    tita[!, :padded_idx] = idx_pdd

    return (tita)
end

function _make_tile(x::Int64, tiles_tab::DataFrames.DataFrame, rast::Rasters.RasterStack)
    tile = rast[X(first(tiles_tab[x, :padded_idx])), Y(last(tiles_tab[x, :padded_idx]))]
    tile[:target_qualities][:, :, 1] .= NaN
    xvals = collect(Rasters.lookup(tile, X))
    yvals = collect(Rasters.lookup(tile, Y))
    xmm_tmp = extrema(collect(first(tiles_tab[x, :dims])))
    ymm_tmp = extrema(collect(last(tiles_tab[x, :dims])))
    xmm = [findnearest(xvals, xmm_tmp[1]), findnearest(xvals, xmm_tmp[2])]
    ymm = [findnearest(yvals, ymm_tmp[1]), findnearest(yvals, ymm_tmp[2])]
    tile[:target_qualities][X(xmm[1]:xmm[2]), Y(ymm[2]:ymm[1])] = rast[:target_qualities][X(first(tiles_tab[x, :idx])), Y(last(tiles_tab[x, :idx]))]
    return (tile)
end

function prune_tile_table(tiles_tab::DataFrames.DataFrame, rast::Rasters.RasterStack)
    keep_rws = zeros(size(tiles_tab)[1])
    for k in (1:size(tiles_tab)[1])
        test = rast[:target_qualities][X(first(tiles_tab[k, :padded_idx])), Y(last(tiles_tab[k, :padded_idx]))]
        test = replace_missing(test, 0.0)
        map!(x -> isnan(x) ? 0.0 : x, test, test)
        test = sum(test[:, :, 1])
        if (test > 0.0)
            keep_rws[k] = 1
        end
    end
    tita = tiles_tab[keep_rws.==1, :]
    return (tita)
end

function tiled_conscape(tiles_tab::DataFrames.DataFrame, outdir, rast::Rasters.RasterStack, radius::Real,
    thetas_jobs::Vector{Dict{String,Any}}, cntr_size::Integer=1; cost_fun::Union{ConScape.Transformation}=nothing)

    for k in (1:size(tiles_tab)[1])
        crp = _make_tile(k, tiles_tab, rast)
        res = moving_window_conscape(crp, radius, thetas_jobs, cntr_size, cost_fun)
        for l in (1:size(thetas_jobs)[1])
            Rasters.write(joinpath(outdir, "Tile" * string(tiles_tab[k, 1]) * "_" * get(thetas_jobs[l], "file_name", "error")), res[l])
        end
    end
end

function tile_progress_map(tiles_tab, datadir)
    fils = readdir(datadir)
    tils = map(x -> split(x, "_")[1], fils)
    fils = fils[occursin.("Tile", tils)]
    tils = tils[occursin.("Tile", tils)]

    tils = map(x -> parse(Int64, replace(x, "Tile" => "")), tils)

    plot(tiles_tab[:, :crds], seriestype=:scatter, mc=:grey, yflip=true, legend=false, ms=2, size=(500, 600),
        title=string(round(Int, sum(map(x -> x ∈ tils, tiles_tab[:, :ID])) / size(tiles_tab)[1] * 100)) * "% completed")
    plot!(tiles_tab[map(x -> x ∈ tils, tiles_tab[:, :ID]), :crds], seriestype=:scatter, mc=:green, ms=5)
end

function stitch_conscape_tiles(datadir, rast::Rasters.RasterStack; outdir=nothing, cleanup=false, mask=true)
    if isnothing(outdir)
        outdir = datadir
    end

    fils = readdir(datadir)
    tils = map(x -> split(x, "_")[1], fils)

    # Select those files that correspond to tiles (in case the folder is polluted with non-tile files) 
    fils = fils[occursin.("Tile", tils)]
    tils = tils[occursin.("Tile", tils)]

    mps = [replace(fils[i], (tils[i] * "_") => "") for i in (1:size(tils)[1])]
    umps = unique(mps)

    xvals_rng = Rasters.lookup(rast, X)
    yvals_rng = Rasters.lookup(rast, Y)
    xvals = collect(xvals_rng)
    yvals = collect(yvals_rng)
    resol = resolution(rast)

    if mask
        msk = Rasters.missingmask(rast[:affinities])
    end

    for i in umps
        mos = Raster(zeros(X(xvals_rng), Y(yvals_rng)))
        if mask
            mos = mos .* msk
        end

        for j in findall(mps .== i)
            tmp = Rasters.Raster(joinpath(datadir, fils[j]))
            xmm_tmp = extrema(collect(Rasters.lookup(tmp, X)))
            ymm_tmp = extrema(collect(Rasters.lookup(tmp, Y)))
            xmm = [findnearest(xvals, xmm_tmp[1]), findnearest(xvals, xmm_tmp[2])]
            ymm = [findnearest(yvals, ymm_tmp[1]), findnearest(yvals, ymm_tmp[2])]
            mos[X(xmm[1]:xmm[2]), Y(ymm[2]:ymm[1])] += tmp
        end

        #mos = Rasters.mosaic(sum, tmp...)
        #plot(mos)
        Rasters.write(joinpath(outdir, i), mos[:, :, 1])
    end
    if cleanup
        for i in fils
            rm(joinpath(datadir, i))
        end
    end
end

function cleanup_conscape_tiles(datadir)
    fils = readdir(datadir)
    tils = map(x -> split(x, "_")[1], fils)

    # Select those files that correspond to tiles (in case the folder is polluted with non-tile files) 
    fils = fils[occursin.("Tile", tils)]
    for i in fils
        rm(joinpath(datadir, i))
    end
end


# Optimization of the moving window computation
function computation_performance_center(rast::Rasters.RasterStack, radius::Real,
    thetas_jobs::Vector{Dict{String,Any}}, cntr_sizes::Vector{Int64}=(1:10), nsample::Int64=10, cost_fun=ConScape.MinusLog())
    df = DataFrames.DataFrame(size=cntr_sizes, n_windows=repeat([0], size(cntr_sizes)[1]), n_samples=repeat([0], size(cntr_sizes)[1]),
        run_time=repeat([now() - now()], size(cntr_sizes)[1]), single_time=repeat([now() - now()], size(cntr_sizes)[1]), total_time=repeat([now() - now()], size(cntr_sizes)[1]))

    for i in (1:size(cntr_sizes)[1])
        cntr_size = cntr_sizes[i]

        xcrds = collect(rast[:affinities].dims[1])
        ycrds = collect(rast[:affinities].dims[2])
        xgrps = ceil.((1:size(xcrds)[1]) ./ cntr_size)
        ygrps = ceil.((1:size(ycrds)[1]) ./ cntr_size)

        cntrs = Any[]

        for i in (1:maximum(xgrps))
            for j in (1:maximum(ygrps))
                push!(cntrs, vec(collect(Iterators.product(xcrds[xgrps.==i], ycrds[ygrps.==j]))))
            end
        end
        df[i, :n_windows] = size(cntrs)[1]

        smpl = ceil(size(cntrs)[1] / nsample)
        smpl = findall(mod.(collect(1:size(cntrs)[1]), smpl) .== 0)
        cntrs = cntrs[smpl]

        strt_tim = now()
        vec_of_stackvecs = [_do_window(cntr, rast, radius, thetas_jobs, cost_fun) for cntr in cntrs]
        res = [_merge_windows(vec_of_stackvecs, rast, thetas_jobs, x) for x in (1:size(thetas_jobs)[1])]
        end_tim = now()

        df[i, :n_samples] = size(cntrs)[1]
        df[i, :run_time] = (end_tim - strt_tim)
        df[i, :single_time] = div((end_tim - strt_tim), size(cntrs)[1])
    end
    df[:, :total_time] = df[:, :single_time] .* df[:, :n_windows]
    return (df)
end

# Core moving window computation
function moving_window_conscape(rast::Rasters.RasterStack, 
                                radius::Real,
                                thetas_jobs::Vector{Dict{String,Any}}, 
                                cntr_size::Integer=1, 
                                cost_fun=nothing, 
                                auto_parallel::Bool=true)

    xcrds = collect(Rasters.lookup(rast, X))
    ycrds = collect(Rasters.lookup(rast, Y))
    xgrps = ceil.((1:size(xcrds)[1]) ./ cntr_size)
    ygrps = ceil.((1:size(ycrds)[1]) ./ cntr_size)

    cntrs = Any[]

    for i in (1:maximum(xgrps))
        for j in (1:maximum(ygrps))
            push!(cntrs, vec(collect(Iterators.product(xcrds[xgrps.==i], ycrds[ygrps.==j]))))
        end
    end

    if (Threads.nthreads() > 1 && auto_parallel)
        vec_of_stackvecs = ThreadsX.map(cntr -> _do_window(cntr, rast, radius, thetas_jobs, cost_fun), cntrs)
    else
        vec_of_stackvecs = [_do_window(cntr, rast, radius, thetas_jobs, cost_fun) for cntr in cntrs]
    end

    res = [_merge_windows(vec_of_stackvecs, rast, thetas_jobs, x) for x in (1:size(thetas_jobs)[1])]
    return (res)
end


# support functions for the moving window computations
function _do_window(cntr, rast::Rasters.RasterStack, radius::Real, thetas_jobs::Vector{Dict{String,Any}}, cost_fun)
    g = _cut_grid(rast, cntr, radius, cost_fun)
    tmp_res = nothing
    if typeof(g) == ConScape.Grid
        try
            tmp = _run_jobs_on_gridwindow(g, thetas_jobs)
            tmp_res = _MatVec2Raster(tmp, rast, cntr, radius)
            #tmp_res = _RastVec2StackVec(tmp_res, thetas_jobs)
        catch x
        end
    end
    return (tmp_res)
end

function _crop(rast::Rasters.RasterStack, ext::ArchGDAL.GDAL.OGREnvelope)
    xvals = collect(Rasters.lookup(rast, X))
    yvals = collect(Rasters.lookup(rast, Y))

    xmm = [findnearest(xvals, ext.MinX), findnearest(xvals, ext.MaxX)]
    ymm = [findnearest(yvals, ext.MinY), findnearest(yvals, ext.MaxY)]

    crp = rast[X(xmm[1]:xmm[2]), Y(ymm[2]:ymm[1])]
    return (crp)
end

function _cut_grid(rast::Rasters.RasterStack, cntr, radius::Real, cost_fun)
    #tgt = ArchGDAL.createpoint()
    tgt = ArchGDAL.createmultipoint()
    for x in 1:size(cntr)[1]
        pt = ArchGDAL.createpoint(cntr[x][1], cntr[x][2])
        ArchGDAL.addgeom!(tgt, pt)
    end
    src = ArchGDAL.buffer(ArchGDAL.centroid(tgt), radius + (resolution(rast)[1] * sqrt(size(cntr)[1])))
    ext = ArchGDAL.envelope(src)
    crp = _crop(rast, ext)
    #plot(crp[:affinities])
    #plot!(src)
    #plot!(tgt)

    crp_pxls = vec(collect(Iterators.product(collect(Rasters.lookup(crp, X)), collect(Rasters.lookup(crp, Y)))))

    # extract quality matrix
    src_mat = Matrix(Array(crp[:source_qualities])[:, :, 1])
    src_mat = map(x -> x < 0 ? NaN : x, src_mat)
    tgt_mat = Matrix(Array(crp[:target_qualities])[:, :, 1])
    tgt_mat = map(x -> x < 0 ? NaN : x, tgt_mat)
    #ConScape.heatmap(src_mat, yflip=true)
    #ConScape.heatmap(tgt_mat, yflip=true)

    X_minmax = [minimum(map(x -> first(x), cntr)), maximum(map(x -> first(x), cntr))]
    Y_minmax = [minimum(map(x -> last(x), cntr)), maximum(map(x -> last(x), cntr))]
    tmp = map(x -> (X_minmax[1] <= first(x) <= X_minmax[2]) & (Y_minmax[1] <= last(x) <= Y_minmax[2]), crp_pxls)
    tmp = reshape(tmp, size(tgt_mat))
    #ConScape.heatmap(tmp, yflip=true)
    tgt_mat = tmp .* tgt_mat

    tmp = sqrt.((sum(map(x -> first(x), cntr)) / size(cntr)[1] .- map(x -> first(x), crp_pxls)) .^ 2 +
                (sum(map(x -> last(x), cntr)) / size(cntr)[1] .- map(x -> last(x), crp_pxls)) .^ 2)
    tmp = tmp .<= radius + (resolution(rast)[1] * sqrt(size(cntr)[1]))
    radius_mat = reshape(tmp, size(src_mat))
    #ConScape.heatmap(radius_mat, yflip=true)


    # extract affinity matrix
    if :affinities in names(rast)
        a_mat = crp[:affinities]
    elseif :source_qualities in names(rast)
        a_mat = crp[:source_qualities]
    else
        a_mat = crp[:qualities]
    end
    a_mat = Matrix(Array(a_mat)[:, :, 1])
    a_mat = map(x -> x < 0 ? NaN : x, a_mat)

    if :costs in names(rast)
        c_mat = crp[:costs]
        c_mat = Matrix(Array(c_mat)[:, :, 1])
        c_mat = map(x -> x < 0 ? NaN : x, c_mat)
    else
        c_mat = ConScape.mapnz(cost_fun, a_mat)
    end

    non_matches = (xor.(isnan.(a_mat), isnan.(src_mat)) +
                   xor.(isnan.(a_mat), isnan.(c_mat)) +
                   xor.(isnan.(c_mat), isnan.(src_mat)))
    non_matches = findall(non_matches .> 0)
    a_mat[non_matches] .= NaN
    c_mat[non_matches] .= NaN
    src_mat[non_matches] .= NaN
    tgt_mat[non_matches] .= NaN
    map!(x -> isnan(x) ? 0 : x, tgt_mat, tgt_mat)
    tgt_mat = sparse(tgt_mat)

    # create grid
    g = silent_Grid(size(src_mat .* radius_mat)...,
        affinities=ConScape.graph_matrix_from_raster(a_mat .* radius_mat),
        source_qualities=radius_mat .* src_mat,
        target_qualities=tgt_mat,
        costs=ConScape.graph_matrix_from_raster(c_mat .* radius_mat))
    if length(nonzeros(tgt_mat)) == 0
        g = NaN
    end
    return (g)
end

function _run_jobs_on_gridwindow(g::ConScape.Grid, thetas_jobs::Vector{Dict{String,Any}})
    resss = Any[]

    for i in 1:size(thetas_jobs)[1]
        ress = Any[]
        h = ConScape.GridRSP(g, θ=get(thetas_jobs[i], "theta", "error"))
        jobs = get(thetas_jobs[i], "jobs", "error")

        for j in 1:size(jobs)[1]
            job = jobs[j]
            job_fun = get(job, "function", "error")
            if (job_fun == "ConScape.connected_habitat")
                res = ConScape.connected_habitat(h; get(job, "kwargs", "error")...)
                map!(x -> isnan(x) ? 0 : x, res, res)
            elseif (job_fun == "ConScape.betweenness_kweighted")
                res = ConScape.betweenness_kweighted(h; get(job, "kwargs", "error")...)
                map!(x -> isnan(x) ? 0 : x, res, res)
            elseif (job_fun == "ConScape.betweenness_qweighted")
                res = ConScape.betweenness_qweighted(h; get(job, "kwargs", "error")...)
                map!(x -> isnan(x) ? 0 : x, res, res)
            else
                throw("This job function is not (yet) implemented")
            end
            push!(ress, res)
        end
        push!(resss, ress)
    end
    return (resss)
end

function _MatVec2Raster(MatVec::Vector{Any}, rast::Rasters.RasterStack, cntr, radius::Real)
    tgt = ArchGDAL.createmultipoint()
    for x in 1:size(cntr)[1]
        pt = ArchGDAL.createpoint(cntr[x][1], cntr[x][2])
        ArchGDAL.addgeom!(tgt, pt)
    end
    src = ArchGDAL.buffer(ArchGDAL.centroid(tgt), radius + (resolution(rast)[1] * sqrt(size(cntr)[1])))
    ext = ArchGDAL.envelope(src)
    crp = _crop(rast, ext)

    RastVec = Any[]

    for i in 1:size(MatVec)[1]
        res = Any[]

        for j in 1:size(MatVec[i])[1]
            tmp = MatVec[i][j] #no transpose is needed
            tmp = reshape(tmp, (size(tmp)..., 1))
            tmp = Rasters.Raster(tmp[:, :, 1], dims(crp)[1:2])
            push!(res, tmp)
        end
        push!(RastVec, res)
    end
    return (RastVec)
end

function _RastVec2StackVec(RastVec::Vector{Any}, thetas_jobs::Vector{Dict{String,Any}})
    # Not used
    StackVec = Any[]
    RastVec = tmp_res

    for i in 1:size(RastVec)[1]
        lyr_nms = map(x -> get(x, "layer_name", "error"), get(thetas_jobs[i], "jobs", "error"))
        lyr_nms = map(x -> Meta.parse(x), lyr_nms)
        tmp = (; zip(lyr_nms, RastVec[i])...)
        tmp = RasterStack(tmp)
        push!(StackVec, tmp)
    end
    return (StackVec)
end

function _merge_windows(vec_of_stackvecs::Union{Vector{Union{Nothing,Vector{Any}}},Vector{Vector{Any}},Vector{Nothing}},
    rast::Rasters.RasterStack, thetas_jobs::Vector{Dict{String,Any}}, x::Integer)

    j = x

    xvals = collect(Rasters.lookup(rast, X))
    yvals = collect(Rasters.lookup(rast, Y))

    lyr_nms = map(x -> get(x, "layer_name", "error"), get(thetas_jobs[j], "jobs", "error"))
    lyr_nms = map(x -> Meta.parse(x), lyr_nms)

    res = map(x -> 0.0, copy(rast[:affinities][:, :, 1]))
    res = [copy(res) for i in (1:size(lyr_nms)[1])]
    res = (; zip(lyr_nms, res)...)
    res = RasterStack(res)

    for i in 1:size(vec_of_stackvecs)[1] #number of windows
        if !(isnothing(vec_of_stackvecs[i]))
            tmp = (; zip(lyr_nms, vec_of_stackvecs[i][j])...)
            tmp = RasterStack(tmp)

            xmm_tmp = extrema(collect(Rasters.lookup(tmp, X)))
            ymm_tmp = extrema(collect(Rasters.lookup(tmp, Y)))
            xmm = [findnearest(xvals, xmm_tmp[1]), findnearest(xvals, xmm_tmp[2])]
            ymm = [findnearest(yvals, ymm_tmp[1]), findnearest(yvals, ymm_tmp[2])]

            for k in 1:size(lyr_nms)[1]
                res[lyr_nms[k]][X(xmm[1]:xmm[2]), Y(ymm[2]:ymm[1])] += tmp[lyr_nms[k]]
            end
            plot(res)
        end
    end
    return (res)
end

