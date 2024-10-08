"""
    Regularizer used to constrain domain to infty-norm ball

    f(x) = ∑ log(2cosh(x_i))
"""
struct SumLogCosh{T<:AbstractFloat, V<:AbstractVector{T}} <: Regularizer
    g::V      # buffer for gradient
end

"""
    Evaluate sum_logcosh, its gradient, and Hessian at `p`:

        f = obj!(slc, p)

    where `p` is a vector of length `n` and `slc` is a `SumLogCosh` object.

    This implementation safeguards against numerical under/overflow.
"""
function obj!(slc::SumLogCosh, p)
    @unpack g = slc
    # Compute the objective function value
    f = sum(logcosh_stable.(p))
    # Compute the gradient
    @. g = tanh(p)
    return f
end

"""
    Numerically stable computation of log(2cosh(x))
"""
function logcosh_stable(x)
    if abs(x) <= 20.0
        return log(2cosh(x))
    else
        # For large x, log(2cosh(x)) ≈ abs(x) + log(1 + e^{-2abs(x)})
        # Since e^{-2abs(x)} is negligible, we can approximate log(1 + e^{-2abs(x)}) ≈ 0
        return abs(x)
    end
end

"""
    Get the gradient of sum_logcosh at the point `p` where
    the `slc` objective was last evaluated:

        g = grad(slc)
"""
grad(slc::SumLogCosh) = slc.g

"""
    Get the Hessian of sum_logcosh at the point `p` where
    the `slc` objective was last evaluated:

        H = hess(slc)
"""
function hess(slc::SumLogCosh)
    @unpack g = slc
    # Compute the Hessian diagonal elements
    h = @. 1 - g^2  # sech^2(p_i) = 1 - tanh^2(p_i)
    return Diagonal(h)
end
