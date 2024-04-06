module KLLS

using LinearAlgebra
using Printf

export newton_opt, Tracer

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
    max_iter::Int = 100,
    μ::Real = 1e-4)

    tracer = Tracer()
    y = copy(y0)
    logΣexp = LogExpFunction(q)
    ls_its = 0

    evaldual(y) = begin
        f, p, H = logΣexp(A'y)
        dObj = f + 0.5λ * dot(y, y) - dot(b, y)
        dGrd = A*p + λ*y - b 
        dHes = A*H*A' + λ*I
        return dObj, dGrd, dHes, p
    end

    dObj, dGrd, dHes, p = evaldual(y)
    ϵ = optTol * (1 + norm(dGrd, Inf)) # relative tolerance

    for k ∈ 0:max_iter

        # Log and test for exit
        logger!(tracer, k, dObj, dGrd, dHes, ls_its)
        if norm(dGrd, Inf) < ϵ
            break
        end

        # Newton step. Replace with a CG solver for large problems
        d = -(dHes \ dGrd)
        if dot(d, dGrd) > 0
            error("no descent")
        end

        # Line search
        α, ls_its = armijo(y->evaldual(y)[1], dGrd, y, d, μ=μ)

        # Update y and evaluate objective quantities
        @. y = y + α*d
        dObj, dGrd, dHes, p = evaldual(y)

    end
    return p, y, tracer
end

function armijo(f, ∇fx, x, d; μ=1e-4, α=1, ρ=0.5, maxits=10)
    for k in 1:maxits
       if f(x+α*d) < f(x) + μ*α*dot(∇fx,d)
           return α, k
       end
       α *= ρ
    end
    error("backtracking linesearch failed")
end

function logger!(tracer, k, dObj, dGrd, dHes, ls_its)
    nrmdGrd = norm(dGrd)
    if k == 0
       @printf("%4s: %11s %11s %8s\n", "iter", "obj", "|grd|", "line its")
    end
    @printf("%4d: %11.4e %11.4e %8d\n", k, dObj, nrmdGrd, ls_its)
    push!(tracer, dObj, nrmdGrd)
end

struct Tracer
    dObj::Vector{Float64} # dual objective
    pResid::Vector{Float64} # primal residuals
end
Tracer() = Tracer(Float64[], Float64[])
import Base: push!
function push!(t::Tracer, dObj, nrmdGrd)
    push!(t.dObj, dObj)
    push!(t.pResid, nrmdGrd)
end

end # module