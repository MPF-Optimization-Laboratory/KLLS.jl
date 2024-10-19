
data = load('data/Laplace_inversion.mat');
A = data.A;

b = data.b;
[m, n] = size(A);
mu = ones(n,1);
p = data.p';
x_plot = data.x_values;

% this choices of d2 gives the standard framework
% i.e no weights for rho-meson problem

for lam =[ 10e-6]

    d2 = sqrt(lam) * ones(m, 1);

    [x, y, z, inform,tracer] = PDCO_KL(A, b, mu, d2);
    name = strcat('rho_meson_pdco_lam',string(lam));

end
