function ecological_distance(grid)
    active_vertices = list_active_vertices(grid)
    A = affinities(grid)
    C = cost_matrix(grid)
    Pref = _Pref(A)
    W = _W(Pref, θ, C)

    W_pruned = W[active_vertices, active_vertices]
    C_pruned = C[active_vertices, active_vertices]

    Z_pruned = inv(Matrix(I - W_pruned))
    C̄_pruned = Z_pruned * ((C_pruned .* W_pruned)*Z_pruned)

    C̄_pruned ./= Z_pruned
    # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
    replace!(C̄_pruned, NaN => Inf)
    dˢ  = diag(C̄_pruned)
    C̄_pruned .-= dˢ'
    return C̄_pruned
end

function calculate_functional_habitat(q, K)
    return sum(q .* (K * q))
end

_Pref(A::SparseMatrixCSC) = Diagonal(inv.(vec(sum(A, dims=2)))) * A

function _W(Pref::SparseMatrixCSC, θ::Real, C::SparseMatrixCSC)

    n = LinearAlgebra.checksquare(Pref)
    if LinearAlgebra.checksquare(C) != n
        throw(DimensionMismatch("Pref and C must have same size"))
    end

    W = Pref .* exp.((-).(θ) .* C)
    replace!(W.nzval, NaN => 0.0)

    return W
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
