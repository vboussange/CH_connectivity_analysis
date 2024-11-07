using SparseArrays
using LinearAlgebra
using GridGraphs
using CUDA.CUSPARSE: AbstractCuSparseArray

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

function dense(sp_mat::M) where M <: AbstractSparseMatrix
    if M <: AbstractCuSparseArray
        return CuArray(sp_mat)
    else
        return Array(sp_mat)
    end
end

function fundamental_matrix(W::AbstractMatrix)
    inv(I - W)
end

function rsp_distance(A::M, θ) where M <: AbstractSparseMatrix
    C = mapnz(A, x -> -log(x)) # cost matrix
    Prw = Diagonal(inv.(vec(sum(A, dims=2)))) * A # random walk probability
    W = Prw .* exp.(- θ .* C)
    Z = fundamental_matrix(dense(W))

    C̄ = Z * ((C .* W)*Z)
    
    C̄ = ifelse.(Z .!= 0, C̄ ./ Z, Inf)
    dˢ = diag(C̄)
    C̄ = C̄ .- dˢ'
    return C̄
end

function rsp_distance(grid::Grid, θ::Real)
    A = affinities(grid) # affinities
    rsp_distance(A, θ)
end

# # here we do not prune the graph, but calculate RSP for `active_vertices`
# function rsp_distance_sparse(A, θ, active_vertices)
#     C = mapnz(A, x->-log(x))
#     Pref = _Pref(A)
#     W = _W(Pref, θ, C)
#     b = Matrix(sparse(active_vertices, active_vertices, one(eltype(A)), size(A)...))
#     Z = sparse((I - W) \ b)
#     C̄ = Z * ((C .* W)*Z)

#     is, js, Zv = findnz(Z)
#     _, _, C̄v = findnz(C̄)

#     C̄ = sparse(is, js, C̄v ./ Zv, size(A)...)
#     # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
#     # C̄ = replace(C̄, NaN => Inf)
#     dˢ  = diag(C̄)
#     C̄ = C̄ .- dˢ'
#     return C̄
# end

# function rsp_distance(A, θ, active_vertices)
#     C = mapnz(A, x->-log(x))
#     Pref = _Pref(A)
#     W = _W(Pref, θ, C)
#     b = Matrix(sparse(active_vertices, active_vertices, one(eltype(A)), size(A)...))
#     Z = (I - W) \ b
#     C̄ = Z * ((C .* W)*Z)

#     C̄ = C̄ ./ (Z .+ eps(eltype(Z)))
#     # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
#     dˢ  = diag(C̄)
#     C̄ = C̄ .- dˢ'
#     return C̄
# end

# function rsp_distance_gpu(A, θ, active_vertices)
#     C = mapnz(A, x->-log(x))
#     Pref = _Pref(A)
#     W = CuArray(_W(Pref, θ, C))
#     b = CuArray(sparse(active_vertices, active_vertices, one(eltype(A)), size(A)...))
#     Z = (I - W) \ b
#     C̄ = Z * ((C .* W)*Z)

#     C̄ = C̄ ./ (Z .+ eps(eltype(Z)))
#     # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
#     dˢ  = diag(C̄)
#     C̄ = C̄ .- dˢ'
#     return C̄
# end

# function rsp_distance(grid::Grid, θ)
#     active_vertices = list_active_vertices(grid)
#     A = affinities(grid)[active_vertices, active_vertices]
#     C = mapnz(A, x->-log(x))
#     Pref = _Pref(A)
#     W = _W(Pref, θ, C)

#     Z = inv(Matrix(I - W))
#     C̄ = Z * ((C .* W)*Z)

#     C̄ ./= Z
#     # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
#     C̄ .= ifelse.(isnan.(C̄), Inf, C̄)
#     dˢ = diag(C̄)
#     C̄ .-= dˢ'
#     return C̄
# end

# function rsp_distance_gpu(grid::Grid, θ)
#     active_vertices = list_active_vertices(grid)
#     A = affinities(grid)[active_vertices, 
#                         active_vertices]
#     A = CuSparseMatrixCSC{Float32}(A)

#     C = mapnz(A, x->-log(x))
#     Pref = _Pref(A)
#     W = CuArray(_W(Pref, θ, C))

#     Z = inv(I - W)
#     C̄ = Z * ((C .* W)*Z)

#     C̄ = C̄ ./ Z
#     # Zeros in Z can cause NaNs in C̄ ./= Z computation but the limit
#     C̄ .= ifelse.(isnan.(C̄), Inf, C̄)
#     dˢ = diag(C̄)
#     C̄ = C̄ .- dˢ'
#     return C̄
# end


# # TODO: to remove
# function connected_habitat(grsp, S::Matrix; diagvalue::Union{Nothing,Real}=nothing)

#     g = _get_grid(grsp)
#     targetidx, targetnodes = _targetidx_and_nodes(g)

#     if diagvalue !== nothing
#         for (j, i) in enumerate(targetnodes)
#             S[i, j] = diagvalue
#         end
#     end

#     qˢ = [g.source_qualities[i] for i in g.id_to_grid_coordinate_list]
#     qᵗ = [g.target_qualities[i] for i in targetidx]

#     funvec = connected_habitat(qˢ, qᵗ, S)

#     func = fill(NaN, g.nrows, g.ncols)
#     for (ij, x) in zip(g.id_to_grid_coordinate_list, funvec)
#         func[ij] = x
#     end

#     return func
# end



# TODO: you probably want to implement this only starting from a grid
# You need to backtest it with ConScape
# function RSP_betweenness_kweighted(W::SparseMatrixCSC,
#     Z::CuSparseMatrixCSR,  # Fundamental matrix of non-absorbing paths
#     qˢ::CuArray, # Source qualities
#     qᵗ::CuArray, # Target qualities
#     S::CuArray,  # Matrix of proximities
#     )

#     Z_inv = @. ifelse(isfinite(Z), inv(Z), floatmax(eltype(Z)))
#     Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

#     KZⁱ = qˢ .* S .* qᵗ'

#     # If any of the values of KZⁱ is above one then there is a risk of overflow.
#     # Hence, we scale the matrix and apply the scale factor by the end of the
#     # computation.
#     λ = max(1.0, maximum(KZⁱ))
#     k = vec(sum(KZⁱ, dims=1)) * inv(λ)

#     KZⁱ .*= inv.(λ) .* Zⁱ
#     diag(KZⁱ) -= k .* diag(Zⁱ)

#     ZKZⁱt = (I - W)'\KZⁱ
#     ZKZⁱt .*= λ .* Z

#     return vec(sum(ZKZⁱt, dims=2)) # diag(Z * KZⁱ')
# end