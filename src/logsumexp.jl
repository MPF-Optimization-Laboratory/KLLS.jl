struct LogExpFunction
    q::Vector{Float64}  # prior
    g::Vector{Float64}  # buffer for gradient
end

"""
Constructor for the LogExpFunction object.

If an n-vector of priors `q` is available:

    LogExpFunction(q)
    
If no prior is known, instead provide the dimension `n`:

    LogExpFunction(n)
"""
function LogExpFunction(q::Vector) 
    @assert all(ζ->ζ≥0, q) && sum(q) ≈ 1
    LogExpFunction(q, similar(q))
end
LogExpFunction(n::Int) = LogExpFunction(fill(1/n, n))

"""
Evaluate logΣexp, its gradient, and Hessian at `p`:

    f, g, H = lse(p)

where `p` is a vector of length `n` and `lse` is a LogExpFunction object.

This implementation safeguards against numerical under/overflow:

https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/4f9b688d298157dc24a5b0a518d971221fbe15dd/src/resampling.jl#L10
"""
function (lse::LogExpFunction)(p::Vector)
    q, g = lse.q, lse.g
    maxval, maxind = findmax(p)
    @. g = q*exp(p - maxval)
    Σ = sum_all_but(g, maxind) # Σ = ∑gₑ-1
    f = log1p(Σ) + maxval
    @. g = g / (Σ+1)
    H = Diagonal(g) - g * g'
    return f, g, H
end

function sum_all_but(w, i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    return s
end
