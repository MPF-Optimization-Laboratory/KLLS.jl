"""
    Regularizer used to constrain domain to one-norm ball

    f(x) = log(2∑cosh(x_i))
"""
struct LogSumCosh{T<:AbstractFloat, V<:AbstractVector{T}} <: Regularizer
    g::V      # buffer for gradient
    e_p::V    # buffer for positive exponentials
    e_n::V    # buffer for negative exponentials
    s::T      # sum of exponentials
end

"""
    Evaluate logΣcosh, its gradient, and Hessian at `p`:

        f = obj!(lsc, p)

    where `p` is a vector of length `n` and `lsc` is a `LogSumCosh` object.

    This implementation safeguards against numerical under/overflow.
"""
function obj!(lsc::LogSumCosh, p)
    @unpack g, e_p, e_n = lsc
    maxval = maximum(abs.(p))

    @. e_p = exp(p - maxval)
    @. e_n = exp(-p - maxval)
    s_p = sum(e_p)
    s_n = sum(e_n)
    s = s_p + s_n
    lsc.s = s

    f = maxval + log(s)
    @. g = (e_p - e_n) / s
    return f
end

"""
    Get the gradient of logΣcosh at the point `p` where
    the `lsc` objective was last evaluated:

        g = grad(lsc)
"""
grad(lsc::LogSumCosh) = lsc.g

"""
    Get the Hessian of logΣcosh at the point `p` where
    the `lsc` objective was last evaluated:

        H = hess(lsc)
"""
function hess(lsc::LogSumCosh)
    @unpack g, e_p, e_n, s = lsc
    h = ((e_p .+ e_n) ./ (2s)) .- (g .^ 2)
    return Diagonal(h) - g * g'
end
