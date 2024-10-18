function [A, b, p, x_values] = produce_Laplace_data(f, K, s_values, n)
    % produce_laplace_data - Generate data for Laplace transforms and moments.
    %
    % Inputs:
    %   - f: A handle to the density function with support on [0, ∞).
    %   - K: A nonnegative integer representing the moments 0,1,...,K.
    %   - s_values: A vector of positive floats used for Laplace transform measurements.
    %   - n: A positive integer for the number of equally spaced samples.
    %
    % Outputs:
    %   - A: (K+1+length(s_values), n) matrix for Laplace transforms and moments.
    %   - b: a (K+1+length(s_values), 1) vector of measurements.
    %   - p: a (n, 1) vector representing the discretized density function (unnormalized).
    %   - x_values: a (n, 1) vector of x values used to evaluate f.

    m = (K + 1) + length(s_values);
    
    % Discretize f using n equally spaced values between 0 and the upper bound
    eps = 1e-8; % the upper limit
    initial_guess = 20; % Initial guess for fsolve
    
    % Define function to find upper bound
    F = @(x) f(x) - eps;
    
    % Use fzero (MATLAB equivalent of fsolve) to find the upper bound
    upper_bound = fzero(F, initial_guess);
    
    % Use the maximum of upper_bound and the max of s_values to ensure a sufficient range
    upper_bound = max([s_values, upper_bound]);
    if upper_bound == max(s_values)
        disp('Using maximum s value as upper bound.');
    end

    x_values = linspace(0, upper_bound, n);
    p = f(x_values); % the unnormalized probability vector

    A = zeros(m, n);

    for k = 0:K
        A(k+1, :) = (x_values .^ k) / n;
    end

    for i = 1:length(s_values)
        s = s_values(i);
        A(i + K + 1, :) = exp(-s * x_values) / n;
    end

    b = A * p';

end
