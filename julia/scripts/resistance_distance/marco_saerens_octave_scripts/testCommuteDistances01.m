
clear  all
format shortE
format compact

mini = realmin * 1000000;
maxi = realmax / 1000000;

%% Doyle and Snell network, page 38 (arxiv pdf) or 48 (book)
%  The numbering of the nodes has been redefined
C = [ maxi,    1,    1, maxi;     % a <- a
         1, maxi,  1/2,    1;     % b <- c
         1,  1/2, maxi,  1/2;     % c <- d 
      maxi,    1,  1/2, maxi ];   % d -> b

% Resulting currents for a unit difference of potential (r = 2)
% source: 1, target: 4
% (16/19) * [ 0,  9/16, 10/16,     0;
%             0,     0,  2/16,  7/16;
%             0,     0,     0, 12/16;
%             0,     0,     0,     0 ];

% [ 0   4.7368e-01   5.2632e-01            0
%   0            0   1.0526e-01   3.6842e-01
%   0            0            0   6.3158e-01
%   0            0            0            0 ]
%
% Intensity of the current (net flow) from source to target: i = 19/16 = 1.1875
%
% Resulting voltages for a unit difference of potential (r = 2)
% and thus for a total source-target electric flow of 19/16
% v = [ 1; 7/16; 6/16; 0 ]
%
% Resulting voltages for a unit total electric flow (r = 2)
% v = [ 0.84211; 0.36842; 0.31579; 0 ]
%
% Optimal value objective function: 0.421053

% Mantrach's network
% C   =   [ maxi,  2,  2,  2,  2,  maxi,  maxi,  maxi,  maxi,  maxi,  maxi;
%           2,  maxi,  2,  2,  2,  maxi,  maxi,  maxi,  maxi,  maxi,  maxi;
%           2,  2,  maxi,  2,  2,  maxi,  maxi,  maxi,  maxi,  maxi,  maxi;
%           2,  2,  2,  maxi,  2,  maxi,  maxi,  maxi,  maxi,  maxi,  maxi;
%           2,  2,  2,  2,  maxi,  1,  1,  maxi,  maxi,  maxi,  maxi;
%           maxi,  maxi,  maxi,  maxi,  1,  maxi,  1,  maxi,  maxi,  maxi,  maxi;
%           maxi,  maxi,  maxi,  maxi,  1,  1,  maxi,  1,  1,  1,  1;
%           maxi,  maxi,  maxi,  maxi,  maxi,  maxi,  1,  maxi,  1,  1,  1;
%           maxi,  maxi,  maxi,  maxi,  maxi,  maxi,  1,  1,  maxi,  1,  1;
%           maxi,  maxi,  maxi,  maxi,  maxi,  maxi,  1,  1,  1,  maxi,  1;
%           maxi,  maxi,  maxi,  maxi,  maxi,  maxi,  1,  1,  1,  1,  maxi ];

C = (C + C')/2; % if a symmetric matrix is needed

[n,n] = size(C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute adjacency matrix elements as inverse of costs 
A = zeros(n,n);
A(C < maxi) = 1 ./ C(C < maxi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the resistance distance, commute time and commute cost
%  for an undirected graph
D1 = commuteDistances01(C);

disp(D1.RD);
disp(D1.CT);
disp(D1.CC);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the average first passage cost from s to t and from t to s, which
%% should be proportional to the resistance distance on an undirected graph
s = 1; t = 4; % source and target
d2 = averageFirstPassageCostDistance01(A,C,t); % distance to t
d3 = averageFirstPassageCostDistance01(A,C,s); % distance to s 

disp(d2(s) + d3(t)) % is equal to the average commute cost between s and t
