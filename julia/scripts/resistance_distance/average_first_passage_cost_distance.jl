#=
Implementation of the average first-passage cost distance taken from matlab script from Marco Saerens.

=# 

using SparseArrays
using LinearAlgebra

function average_first_passage_cost(A::AbstractMatrix, C::AbstractMatrix, t::Int)
    # INPUT:
    # A: adjacency matrix of a strongly connected graph
    # C: cost matrix
    # t: target node
    #
    # OUTPUT:
    # afpc: Vector containing directed average first-passage costs from each node to target node t

    # Argument checks
    n, m = size(A)
    if n != m
        throw(ArgumentError("The adjacency matrix is not square."))
    end
    if t < 1 || t > m
        throw(ArgumentError("The target node index t is out of bounds."))
    end
    if A != A'
        throw(ArgumentError("The adjacency matrix is not symmetric."))
    end

    e = ones(n)

    # Diagonal matrices of degree and inverse degree
    d = A * e
    if any(d .== 0)
        throw(ArgumentError("The graph has nodes with zero degree, which violates assumptions."))
    end
    d_inv = e ./ d
    P = A .* d_inv'
    P[t, :] .= 0
    pc = (P .* C) * e
    pc[t] = 0
    afpc = (I - P) \ pc
    return afpc
end