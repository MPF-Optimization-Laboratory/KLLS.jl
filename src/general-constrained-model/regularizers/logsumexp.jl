"""
    Regularizer used to constrain domain to infty-norm ball

    f(x) = log(∑exp(x_i))
"""
struct LogSumExp{T<:AbstractFloat, V1<:AbstractVector{T}, V2<:AbstractVector{T}} <: Regularizer
    g::V1 # buffer for gradient
end

"""
    Evaluate logΣexp, its gradient, and Hessian at `p`:

        f = obj!(lse, p)

    where `p` is a vector of length `n` and `lse` is a LogExpFunction object.

    This implementation safeguards against numerical under/overflow:

    https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/4f9b688d298157dc24a5b0a518d971221fbe15dd/src/resampling.jl#L10
"""
function obj!(lse::LogSumExp, p)
    @unpack g = lse
    maxval, maxind = findmax(p)
    @. g = q * exp(p - maxval)
    Σ = sum_all_but(g, maxind) # Σ = ∑gₑ-1
    f = log1p(Σ) + maxval
    @. g = g / (Σ + 1)
    return f
end

"""
Get the gradient of logΣexp at the point `p` where
the `lse` objective was last evaluated:

    g = grad(lse)
"""
grad(lse::LogSumExp) = lse.g

"""
Get the Hessian of logΣexp at the point `p` where
the `lse` objective was last evaluated:

    H = hess(lse)
"""
function hess(lse::LogSumExp)
    g = lse.g
    return Diagonal(g) - g * g'
end

"""
Special all-but-i sum used by logΣexp.
"""
function sum_all_but(w, i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    return s
end