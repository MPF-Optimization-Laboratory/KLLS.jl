% scratch


data = load('data/synthetic-UEG_testproblem.mat');

A = data.A;
[m,n] = size(A)

b = data.b_avg;
x0 = data.x; % already normalized
x0 = x0'; 
b = b';
mu = data.mu; % prior provided
mu = double(mu'); % was single precision
lam = 1e-4

% UEG Weighted 2-norm of residual
% double check this is correct!!
d2 = sqrt(lam)*sqrt(data.b_std');
%d2 = sqrt(lam)*ones(m,1)

[x, y, z, inform,tracer_UEG] = PDCO_KL(A, b, mu, d2);


xsum = sum(x);
disp('Synthetic UEG');
disp('Sum of entries of x:')
disp(xsum); 
disp('Res:')
disp(norm(A*x - b));
disp('RMSE:')
disp(norm(x - x0)/sqrt(n));

x_ = 1:n;
figure
plot(x_ ,x,'g-',x_, mu,'b', x_, x0,'r')
legend('recovered', 'prior', 'truth')

figure
plot(x_,x)