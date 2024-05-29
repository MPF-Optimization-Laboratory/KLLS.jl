struct LogExpFunction{V <: AbstractVector}
    q::V  # prior
    g::V  # buffer for gradient
end

"""
Constructor for the LogExpFunction object.

If an n-vector of priors `q` is available:

    LogExpFunction(q)
    
If no prior is known, instead provide the dimension `n`:

    LogExpFunction(n)
"""
function LogExpFunction(q::AbstractVector) 
    @assert (all(ζ->ζ≥0, q) && sum(q) ≈ 1) "prior is not on the simplex"
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
function obj!(lse::LogExpFunction, p)
    @unpack q, g = lse
    maxval, maxind = findmax(p)
    @tullio g[j] = q[j]*exp(p[j] - maxval)
    Σ = sum_all_but(g, maxind) # Σ = ∑gₑ-1
    f = log1p(Σ) + maxval
    @tullio g[j] = g[j]/(Σ+1) # g = g/(Σ+1)
    return f
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

"""
Structure for KLLS data
    - `A` is the matrix of constraints
    - `b` is the right-hand side
    - `q` is the prior (default: uniform)
    - `lse` is the log-sum-exp function
    - `λ` is the regularization parameter
    - `w` is an n-buffer for the Hessian-vector product 
    - `bNrm` is the norm of the right-hand side
    - `name` is the name of the problem
"""

@kwdef mutable struct KLLSData{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    A::M
    b::V
    q::V = fill(eltype(A)(1/size(A, 2)), size(A, 2))
    lse::LogExpFunction{V} = LogExpFunction(q)
    λ::T = √eps(eltype(A))
    w::V = similar(q)
    bNrm::T = norm(b)
    name::String = ""
end

KLLSData(A, b; kwargs...) = KLLSData(A=A, b=b; kwargs...)

function Base.show(io::IO, data::KLLSData)
    if data.name != ""
        @printf(io, "KLLS data: %s\n", data.name)
    else
        println(io, "KLLS data")
    end
    @printf(io, "size: m = %d, n = %d\n", size(data.A)...)
    @printf(io, "%-8s = %9.2e\n", "norm(b)", norm(data.b))
    @printf(io, "%-8s = %9.2e\n", "sum(b)", sum(data.b))
    @printf(io, "%-8s = %9.2e\n", "λ", data.λ)
end

function dObj!(data::KLLSData{T}, y::V) where {T<:AbstractFloat, V<:AbstractVector{T}}
    @unpack A, b, λ, w, lse = data
    mul!(w, A', y)
    f = obj!(lse, w)
    return f + 0.5λ * y⋅y - b⋅y
end

function dGrad!(data::KLLSData{T}, y::V, ∇f::V) where {T<:AbstractFloat, V<:AbstractVector{T}}
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

function dHess_prod!(data::KLLSData{T}, y::V, v::V) where {T<:AbstractFloat, V<:AbstractVector{T}}
    @unpack A, λ, w, lse = data
    # H = hess(lse)
    # v .= (A*(H*(A')*y)) + λ*y
    g = grad(lse)
    mul!(w, A', y)       # w =            A'y
    w .= g.*(w .- (g⋅w)) # w =  (G - gg')(A'y)
    mul!(v, A, w)        # v = A(G - gg')(A'y)
    if λ > 0
        v .+= λ*y
    end
    return v
end
