lambdas = logspace(-2, 2, 3);
print = 0;
delta = 1e-3;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = load('data/synthetic-UEG_testproblem.mat');

A = data.A;

b = data.b_avg;
b = b';
mu = data.mu; % prior provided
mu = double(mu'); % was single precision

for lam = lambdas
    [x, y, z, inform,tracer_UEG] = PDCO_KL(A, b, mu, lam, delta, print);
    name = strcat('UEG_PDCO_lam',string(lam));
    writematrix(tracer_UEG,strcat(fullfile(directory,name),".csv"))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% rho-meson test problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear A b mu;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = load('data/rho-meson_testproblem.mat');
A = data.A;

b = data.b_avg;
b =  b';
mu = data.mu; % prior provided
mu = double(mu');

for lam = lambdas
    [x, y, z, inform,tracer_rho_meson] = PDCO_KL(A, b, mu, lam, delta, print);
    name = strcat('PDCO_rho_meson_lam',string(lam));
    writematrix(tracer_rho_meson,strcat(fullfile(directory,name),".csv"))
end
