lambdas = logspace(-4, 2, 7)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% UEG Test Problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


data = load('data/synthetic-UEG_testproblem.mat');

A = data.A;
[m, n] = size(A);

b = data.b_avg;
b = b';
x0 = data.x; % already normalized
x0 = x0'; 
mu = data.mu; % prior provided
mu = double(mu'); % was single precision

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% rho-meson test problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for lam = lambdas
    [x, y, z, inform,tracer_UEG] = PDCO_KL(A, b, mu, lmda_test, delta, print);
    t = datetime
    name = strcat('PDCO_UEG',string(month(t)),string(day(t)),'_',string(hour(t)),string(minute(t)),'_',string(lam))
    writematrix(tracer_UEG,fullfile("GitHub\KLLS.jl\numerics\outputs",name,".csv"))
end

data = load('data/rho-meson_testproblem.mat');
A = data.A;
b = data.b_avg;
x0 = data.x;
mu = data.mu; % prior provided
