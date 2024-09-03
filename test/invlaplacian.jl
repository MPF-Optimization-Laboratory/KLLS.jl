"""
Implements methods to generate data for Laplace transforms on probability distributions. This test was suggested by

Udo Wagner; Alois L. J. Geyer
Biometrika, Vol. 82, No. 4. (Dec., 1995), pp. 887-892.
"""

using Distributions
using Roots
using Plots

"""
    gen_laplace_data(f, K, s_values, n)

Generate data for Laplace transforms and moments.

Inputs:
- `f`: A handle to the density function with support on [0, ∞).
- `K`: A nonnegative integer representing the moments 0,1,...,K.
- `svals`: A vector of positive floats used for Laplace transform measurements.
- `n`: A positive integer for the number of equally spaced samples.

Outputs:
- `A`: (K+1+length(s_values), n) matrix for Laplace transforms and moments.
- `b`: A (K+1+length(s_values), 1) vector of measurements.
- `p`: A (n, 1) vector representing the discretized density function (unnormalized).
- `x_values`: A (n, 1) vector of x values used to evaluate `f`.
"""
function gen_Laplace_data(f::Function, K::Int, svals::AbstractVector, n::Int; fmin=sqrt(eps()))

    m = K + 1 + length(svals)
    
    # Define function to find upper bound
    F(x) = f(x) - fmin
    
    # Use fzero (MATLAB equivalent of fsolve) to find the upper bound
    initial_guess = 20 # Initial guess for fsolve
    upper_bound = fzero(F, initial_guess)
    
    # Use the maximum of upper_bound and the max of svals to ensure a sufficient range
    upper_bound = max(svals..., upper_bound)
    if upper_bound == maximum(svals)
        warning("Using maximum s value as upper bound.")
    end

    xvals = range(0, stop=upper_bound, length=n)
    p = f.(xvals) # the unnormalized probability vector

    A = zeros(m, n)

    for k in 0:K
        A[k+1, :] = (xvals .^ k) / n
    end

    for (i, s) in enumerate(svals)
        A[i + K + 1, :] = exp.(-s * xvals) / n
    end

    b = A * p

    return A, b, p, xvals
end

f_LogNormal = x->pdf(LogNormal(0, 1), x)
f_Pareto = x->pdf(Pareto(2, 1), x)
f_Gamma = x->pdf(Gamma(2, 1), x)
f_InvGauss = x->pdf(InverseGaussian(1, 3), x)
x = 10

plot(
    plot(f_LogNormal, 0, 10, label="LogNormal"),
    plot(f_Pareto, 0, 5, label="Pareto"),
    plot(f_Gamma, 0, 5, label="Gamma"),
    plot(f_InvGauss, 0, 10, label="Inverse Gaussian"),
    layout=(2, 2), legend=true
)

K = 2;
n = 100;
s_values = [1, 5, 10];

A, b, fvals, xvals = gen_Laplace_data(f_LogNormal, K, s_values, n)

plot(xvals, fvals, marker=:circle, label="LogNormal")
plot!(f_LogNormal, 0, 10)