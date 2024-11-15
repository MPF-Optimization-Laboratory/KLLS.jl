mutable struct ExecutionStats{T<:AbstractFloat, V<:AbstractVector{T}, S<:AbstractArray{T}, DF}
    status::Symbol
    elapsed_time::T
    iter::Int
    neval_jprod::Int
    neval_jtprod::Int
    primal_obj::T
    dual_obj::T
    solution::S
    residual::V
    optimality::T
    tracer::DF
end

function Base.show(io::IO, s::ExecutionStats)
    rel_res = norm(s.residual, Inf)
    @printf(io, "\n")
    if s.status == :max_iter 
        @printf(io, "Maximum number of iterations reached\n")
    elseif s.status == :optimal
        @printf(io, "Optimality conditions satisfied\n")
    end
    nprods = s.neval_jprod + s.neval_jtprod
    @printf(io, "Products with A and A': %9d\n"  , nprods)
    @printf(io, "Time elapsed (sec)    : %9.1f\n", s.elapsed_time)
    @printf(io, "||Ax-b||₂             : %9.1e\n", rel_res)
    @printf(io, "Optimality            : %9.1e\n", s.optimality)
end

"""
    randKLmodel(m, n)

Generate a random KL model of size `m` x `n`.
"""
function randKLmodel(m, n)
    A = randn(m, n)
    b = randn(m)
    return KLLSModel(A, b)
end

"""
    histogram(s:ExecutionStats; kwargs...)

Plot a histogram of the solution.
"""
function histogram end

"""
    value!(kl::KLLSModel, t; kwargs...)

Compute the dual objective of a KLLS model with respect to the scaling parameter `t`.
"""
function value2!(kl::KLLSModel, t; jprods=Int[0], kwargs...)
    @unpack λ, A = kl
    scale!(kl, t)
    s = solve!(kl; kwargs...)
    y = s.residual/λ
    dv = obj!(kl.lse, A'y) - log(t) - 1
    jprods[1] += neval_jprod(kl) + neval_jtprod(kl)
    return dv
end

"""
    maximize!(kl::KLLSModel; kwargs...) -> t, xopt, jprods

TODO: Documentation incomplete and incorrect options
Keyword arguments:
- `t::Real=1.0`: Initial guess for the scaling parameter (root finding)
- `rtol::Real=1e-6`: Relative tolerance for the optimization.
- `atol::Real=1e-6`: Absolute tolerance for the optimization.
- `xatol::Real=1e-6`: Absolute tolerance for the primal solution.
- `xrtol::Real=1e-6`: Relative tolerance for the primal solution.
- `δ::Real=1e-2`: Tolerance for the dual objective.
- `zverbose::Bool=true`: Verbosity flag.
- `logging::Int=0`: Logging level.

Maximize the dual objective of a KLLS model with respect to the scaling parameter `t`.
Returns the optimal primal solution.
"""
function maximize!(
    kl::KLLSModel{T};
    t=one(T),
    rtol=1e-6,
    atol=1e-6,
    xatol=1e-6,
    xrtol=1e-6,
    δ=1e-2,
    zverbose=true,
    logging=0,
    ) where T

    jprods = Int[0]
    dv!(t) = value2!(kl, t; jprods=jprods, atol=δ*atol, rtol=δ*rtol, logging=logging)
    t = Roots.find_zero(dv!, t; atol=atol, rtol=rtol, xatol=xatol, xrtol=xrtol, verbose=zverbose)
    return t, t*grad(kl.lse), jprods[1]
end