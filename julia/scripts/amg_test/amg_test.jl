A = poisson(100); 
b = rand(100);
solve(A, b, RugeStubenAMG(), maxiter = 1, abstol = 1e-6)