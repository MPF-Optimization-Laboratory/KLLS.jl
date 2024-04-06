module KLLS

using LinearAlgebra
using Printf
export newton_opt

struct LogExpFunction
    q::Vector{Float64}  # prior
    g::Vector{Float64}  # buffer for gradient
end

"""
Constructor for the LogExpFunction struct with prior `q`. 

    LogExpFunction(q) -> LogExpFunction(q, similar(q))
"""
function LogExpFunction(q::Vector) 
    @assert all(ζ->ζ≥0, q) && sum(q) ≈ 1
    LogExpFunction(q, similar(q))
end

"""
Constructor for the LogExpFunction struct with uniform prior.

    LogExpFunction(n) -> LogExpFunction(fill(1/n, n))
"""
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

"""
Newton's method for minimizing unconstrained optimization problems

    min_x KL(p|q) + λ/2 ||y||² subj to Ap + λq = b
    min_y logΣexp(A'y) - ⟨b, y⟩ +  λ/2 ||y||²

# Input
- `A` is an m-by-n matrix
- `b` is an m-vector
- `q` is an n-vector of priors in the unit simplex
- `λ` is a nonnegative scalar (default: 1e-6)
- `y0` is the initial guess (default: zeros(m))
- `optTol` is the relative tolerance for the gradient (default: 1e-6)
- `max_iter` is the maximum number of iterations (default: 100)

# Output
- `(p, y)` approximate primal-dual pair
"""
function newton_opt(
    A::Matrix, b::Vector, q::Vector, λ::Real=1e-6;
    y0::Vector = zeros(size(A, 1)),
    optTol::Real = 1e-6,
    max_iter::Int = 100)

    y = copy(y0)
    logΣexp = LogExpFunction(q)

    evaldual(y) = begin
        f, g, H = logΣexp(A'y)
        dObj = f + 0.5λ * dot(y, y) - dot(b, y)
        dGrd = A*g - b + λ*y 
        dHes = A*H*A' + λ*I
        return dObj, dGrd, dHes
    end

    dObj, dGrd, dHes = evaldual(y)
    ϵ = optTol * (1 + norm(dGrd)) # relative tolerance

    for k ∈ 1:max_iter
        logger(k, dObj, dGrd, dHes) 
        norm(dGrd) < ϵ && break
        d = -(dHes \ dGrd)
        if dot(d, dGrd) > 0
            error("no descent")
        end
        @. y = y + d
        dObj, dGrd, dHes = evaldual(y)
    end
    return y
end

function logger(k, dObj, dGrd, dHes)
    if k == 1
       @printf("%4s: %11s %11s %11s\n", "iter", "obj", "|grd|", "|hes|")
    end
    @printf("%4d: %11.4e %11.4e %11.4e\n", k, dObj, norm(dGrd), eigvals(dHes)[1])
end

end # module