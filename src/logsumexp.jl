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
function obj!(lse::LogExpFunction, p::Vector)
    @unpack q, g = lse
    maxval, maxind = findmax(p)
    @tullio g[j] = q[j]*exp(p[j] - maxval)
    Σ = sum_all_but(g, maxind) # Σ = ∑gₑ-1
    f = log1p(Σ) + maxval
    @tullio g[j] = g[j]/(Σ+1) # g = g/(Σ+1)
    return f
end


function sum_all_but(w, i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    return s
end

"""
Get the gradient of logΣexp at the point `p` where
the `lse` objective was last evaluated:

    g = grad(lse)
"""
grad(lse::LogExpFunction) = lse.g

"""
Get the Hessian of logΣexp at the point `p` where
the `lse` objective was last evaluated:

    H = hess(lse)
"""
function hess(lse::LogExpFunction)
    g = lse.g
    return Diagonal(g) - g * g'
end

#=
KLLSData - data structure and methods for dual objective
=#
mutable struct KLLSData{T<:Real}
    A::AbstractMatrix{T}
    b::Vector{T}
    q::Vector{T}
    lse::LogExpFunction
    λ::T
    w::Vector{T} # n-buffer for Hessian-vector product
end

function KLLSData(A, b; kwargs...)
    n = size(A, 2)
    q = fill(1/n, n)
    KLLSData(A, b, q; kwargs...)
end

function KLLSData(A, b, q; λ=1e-6)
    KLLSData(A, b, q, LogExpFunction(q), λ, similar(q))
end

function dObj!(data::KLLSData, y)
    @unpack A, b, λ, w, lse = data
    mul!(w, A', y)
    f = obj!(lse, w)
    return f + 0.5λ * y⋅y - b⋅y
end

function dGrad!(data::KLLSData, y, ∇f) 
    @unpack A, b, λ, lse = data
    p = grad(lse)
    @tullio ∇f[i] = λ*y[i] - b[i]
    mul!(∇f, A, p, 1, 1)
    return ∇f
end

function dHess(data::KLLSData)
    @unpack A, λ, lse = data
    H = hess(lse)
    ∇²dObj = A*H*A'
    if λ > 0
        ∇²dObj += λ*I
    end
    return ∇²dObj
end

function dHess_prod!(data::KLLSData, y, v)
    @unpack A, λ, w, lse = data
    # H = hess(lse)
    # v .= (A*(H*(A')*y)) + λ*y
    g = grad(lse)
    mul!(w, A', y)       # w =            A'y
    w .= g.*w .- g*(g⋅w) # w =  (G - gg')(A'y)
    mul!(v, A, w)        # v = A(G - gg')(A'y)
    if λ > 0
        v .+= λ*y
    end
    return v
end
