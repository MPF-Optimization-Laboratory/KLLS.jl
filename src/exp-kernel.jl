struct ExpKernel{T<:AbstractFloat, V1<:AbstractVector{T}, V2<:AbstractVector{T}}
    q::V1  # prior
    g::V2  # buffer for gradient
end

"""
Constructor for the ExpKernel object.

If an n-vector of priors `q` is available:

    ExpKernel(q)
    
If no prior is known, instead provide the dimension `n`:

    ExpKernel(n)
"""
function ExpKernel(q::AbstractVector)
    @assert (all(ζ -> ζ ≥ 0, q)) "prior is not nonnegative"
    ExpKernel(q, similar(q))
end

"""
Evaluate ∑exp, its gradient, and Hessian at `p`:

    f = obj!(expk, p)

where `p` is a vector of length `n` and `expk` is a ExpKernel object.
"""
function obj!(expk::ExpKernel{T}, p) where T
    @unpack q, g = expk
    @. g = q .* exp(p)
    @. g = g ./ exp(one(T))
    return sum(g)
end

"""
Get the gradient of ∑exp at the point `p` where
the `expk` objective was last evaluated:

    g = grad(expk)
"""
grad(expk::ExpKernel) = expk.g

"""
Get the Hessian of ∑exp at the point `p` where
the `expk` objective was last evaluated:

    H = hess(expk)
"""
function hess(expk::ExpKernel)
    g = expk.g
    return Diagonal(g)
end

"""
Get the Hessian vector product of logΣexp at the point `p`
where the `expk` objective was last evaluated:

    Hz = hessvp!(expk, z)
"""
function hessvp!(expk::ExpKernel{T}, z::AbstractVector{T}) where T
    g = expk.g
    z .= g.*z
    return z
end