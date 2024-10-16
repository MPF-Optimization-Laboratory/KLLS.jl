%%%
% Warning: this relies on altered versions of pdco, KL.m, and PDCO_KL.m
% Previous versions by Nick (prior to 10/15/2024) will not run properly.
% See comments in pdco / pdco_kl for details of changes
%%%

lambdas = logspace(-2, 2, 3);
t = datetime;

folder = strcat(string(month(t)),string(day(t)),'_',string(hour(t)),string(minute(t)));
directory = fullfile("GitHub\KLLS.jl\numerics\outputs",folder);
    
if ~exist(directory, 'dir')
       mkdir(directory)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% UEG Test Problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear A b mu;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = load('data/synthetic-UEG_testproblem.mat');

A = data.A;

b = data.b_avg;
b = b';
mu = data.mu; % prior provided
mu = double(mu'); % was single precision

% UEG Weighted 2-norm of residual
% double check this is correct!!


for lam = lambdas
    d2 = sqrt(lam)*sqrt(data.b_std');

    [x, y, z, inform,tracer_UEG] = PDCO_KL(A, b, mu, d2);
    name = strcat('UEG_pdco_lam',string(lam));
    writematrix(tracer_UEG,strcat(fullfile(directory,name),".csv"))
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% rho-meson test problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear A b mu d2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = load('data/rho-meson_testproblem.mat');
A = data.A;

b = data.b_avg;
b =  b';
mu = data.mu; % prior provided
mu = double(mu');

[m, n] = size(A);

% this choices of d2 gives the standard framework
% i.e no weights for rho-meson problem

for lam = lambdas

    d2 = sqrt(lam) * ones(m, 1);

    [x, y, z, inform,tracer_rho_meson] = PDCO_KL(A, b, mu, d2);
    name = strcat('rho_meson_pdco_lam',string(lam));

    writematrix(tracer_rho_meson,strcat(fullfile(directory,name),".csv"))
end
