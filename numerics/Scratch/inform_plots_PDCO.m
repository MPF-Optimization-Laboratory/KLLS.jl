
data = load('data/rho-meson_testproblem.mat');
A = data.A;

b = data.b_avg';
[m, n] = size(A);
mu = ones(n,1);
p = data.x';
% x_plot = data.x_values;

% this choices of d2 gives the standard framework
% i.e no weights for rho-meson problem
lams = logspace(-12,-2,11);
result_cg = zeros('like',lams);
result_norm = zeros('like',lams);

index = 1;
for lam = lams
    clear tracer x y z 

    d2 = sqrt(lam) * ones(m, 1);

    [x, y, z, inform,tracer] = PDCO_KL(A, b, mu, d2);
    name = strcat('rho_meson_pdco_lam',string(lam));
    

    result_cg(index) = sum(tracer(:,end))
    result_norm(index) = norm(A*x-b)

    index = index +1;
end
