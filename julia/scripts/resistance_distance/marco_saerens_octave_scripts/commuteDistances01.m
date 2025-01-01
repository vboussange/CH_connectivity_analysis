function D = commuteDistances01(C)
% INPUT:
% C: cost matrix
%
% OUTPUT:
% D, a structure containing the matrices of commute-time, commute-cost
% and resistance distances

maxi = realmax / 1000000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Check arguments
% Check if square matrix
[n, m] = size(C);
if n ~= m
    error('The adjacency matrix is not square.')
end

% Check if symmetric matrix / graph is undirected
if ~isequal(C, C')
    error('The cost matrix is not symmetric.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute adjacency matrix elements as inverse of costs 
A = zeros(n,n);
A(C < maxi) = 1 ./ C(C < maxi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Utilities
e = ones(n,1);
E = ones(n,n);
I = eye(n,n);

% Diagonal matrices of degree and inverse degree
d = A*e;
Diag_d = diag(d);
Diag_d_inv = diag(1 ./ d);

% Volume of the graph
vol = sum(d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of the average commute-time, commute-cost and resistance distances
% Laplacian matrix
L = Diag_d - A;
% Transition probability matrix
P0 = A ./ d;

% Pseudoinverse of the Laplacian matrix
L_plus = ((L - (E/n))^(-1)) + (E/n);

% Average commute-time distance
diag_L_plus = diag(L_plus);
RD = (diag_L_plus * e' + e * diag_L_plus' - 2*L_plus);
RD(isnan(RD)) = 0;
RD(isinf(RD)) = maxi;
D.RD = RD;

CT = vol * RD;
D.CT = CT;

D.CC = (e'*(A.*C)*e) * RD;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test by computing the effective resistance between node 1 and last node n
%  This quantity should be equal to the resistance distance
sources = zeros(n,1);
sources(1) = 1; sources(n) = -1;  % unit source injected in (and out)
% the network
v = L_plus * sources; % compute centered potential on nodes
v = v - min(v); % set potential at output node n to 0
disp(v(1)); % effective resistance between node 1 and n

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
