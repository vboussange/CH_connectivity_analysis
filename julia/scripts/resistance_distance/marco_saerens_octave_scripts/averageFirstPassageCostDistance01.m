function afpc = averageFirstPassageCostDistance01(A,C,t)
% INPUT:
% A: adjacency matrix of a strongly connected graph
% C: cost matrix
% t: target node
%
% OUTPUT:
% afpt: contains the vector of directed average first-passage costs from
% each node to target node t

%% Check of arguments
% Check if square matrix
[n, m] = size(A);
if n ~= m
    error('The adjacency matrix is not squared.')
end

if t > m || t < 1
    error('The index of target node t is too large or too small.')
end

% Check if symmetric matrix / graph is undirected
if ~isequal(A, A')
    error('The adjacency matrix is not symmetric.')
end

%% Utilities
e = ones(n,1);
I = eye(n,n);
myMax = 0.9 * realmax;

% Diagonal matrices of degree and inverse degree
d = A*e;;
d_inv = e ./ d;
% Transition probabilility matrix
P = d_inv .* A;

% Computation of the average first-passage time
P(t,:) = 0;
pc = (P .* C) * e;
pc(t) = 0;
afpc = (I - P) \ pc;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
