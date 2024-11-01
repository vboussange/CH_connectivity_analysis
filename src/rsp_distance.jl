using SparseArrays
using LinearAlgebra
function rsp_distance(A, θ)
    C = map(x -> ifelse(x > 0, -log(x), zero(eltype(A))), A)
    Pref = _Pref(A)
    W = _W(Pref, θ, C)

    Z = inv(Matrix(I - W))
    C̄ = Z * ((C .* W)*Z)

    C̄ = C̄ ./ Z
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    # C̄ = replace(C̄, NaN => Inf)
    dˢ  = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance_gpu(A, θ)
    C = mapnz(A, x->-log(x))
    Pref = _Pref(A)
    W = _W(Pref, θ, C)

    W = CuArray(_W(Pref, θ, C))

    Z = inv(I - W)
    C̄ = Z * ((C .* W)*Z)

    C̄ = C̄ ./ Z
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    C̄ .= ifelse.(isnan.(C̄), Inf, C̄)
    dˢ  = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance_sparse(A, θ, active_vertices)
    C = mapnz(A, x->-log(x))
    Pref = _Pref(A)
    W = _W(Pref, θ, C)
    b = Matrix(sparse(active_vertices, active_vertices, one(eltype(A)), size(A)...))
    Z = sparse((I - W) \ b)
    C̄ = Z * ((C .* W)*Z)

    is, js, Zv = findnz(Z)
    _, _, C̄v = findnz(C̄)

    C̄ = sparse(is, js, C̄v ./ Zv, size(A)...)
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    # C̄ = replace(C̄, NaN => Inf)
    dˢ  = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance(A, θ, active_vertices)
    C = mapnz(A, x->-log(x))
    Pref = _Pref(A)
    W = _W(Pref, θ, C)
    b = Matrix(sparse(active_vertices, active_vertices, one(eltype(A)), size(A)...))
    Z = (I - W) \ b
    C̄ = Z * ((C .* W)*Z)

    C̄ = C̄ ./ (Z .+ eps(eltype(Z)))
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    dˢ  = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance_gpu(A, θ, active_vertices)
    C = mapnz(A, x->-log(x))
    Pref = _Pref(A)
    W = CuArray(_W(Pref, θ, C))
    b = CuArray(sparse(active_vertices, active_vertices, one(eltype(A)), size(A)...))
    Z = (I - W) \ b
    C̄ = Z * ((C .* W)*Z)

    C̄ = C̄ ./ (Z .+ eps(eltype(Z)))
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    dˢ  = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance(grid::Grid, θ)
    active_vertices = list_active_vertices(grid)
    A = affinities(grid)[active_vertices, active_vertices]
    C = mapnz(A, x->-log(x))
    Pref = _Pref(A)
    W = _W(Pref, θ, C)

    Z = inv(Matrix(I - W))
    C̄ = Z * ((C .* W)*Z)

    C̄ ./= Z
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    C̄ .= ifelse.(isnan.(C̄), Inf, C̄)
    dˢ = diag(C̄)
    C̄ .-= dˢ'
    return C̄
end

function rsp_distance_gpu(grid::Grid, θ)
    active_vertices = list_active_vertices(grid)
    A = affinities(grid)[active_vertices, 
                        active_vertices]
    A = CuSparseMatrixCSC{Float32}(A)
    
    C = mapnz(A, x->-log(x))
    Pref = _Pref(A)
    W = CuArray(_W(Pref, θ, C))

    Z = inv(I - W)
    C̄ = Z * ((C .* W)*Z)

    C̄ = C̄ ./ Z
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    C̄ .= ifelse.(isnan.(C̄), Inf, C̄)
    dˢ  = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function calculate_functional_habitat(q, K)
    return sum(q .* (K * q))
end

_Pref(A) = Diagonal(inv.(vec(sum(A, dims=2)))) * A

function _W(Pref, θ, C)

    W = Pref .* exp.(- θ .* C)
    # replace!(W.nzval, NaN => 0.0)

    return W
end

function mapnz(mat::M, f) where M <: AbstractSparseMatrix
    I, J, V = findnz(mat)
    return sparse(I, J, f.(V), size(mat)...)
end


# TODO: to remove
function connected_habitat(grsp, S::Matrix; diagvalue::Union{Nothing,Real}=nothing)

    g = _get_grid(grsp)
    targetidx, targetnodes = _targetidx_and_nodes(g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            S[i, j] = diagvalue
        end
    end

    qˢ = [g.source_qualities[i] for i in g.id_to_grid_coordinate_list]
    qᵗ = [g.target_qualities[i] for i in targetidx]

    funvec = connected_habitat(qˢ, qᵗ, S)

    func = fill(NaN, g.nrows, g.ncols)
    for (ij, x) in zip(g.id_to_grid_coordinate_list, funvec)
        func[ij] = x
    end

    return func
end
