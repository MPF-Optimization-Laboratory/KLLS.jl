mutable struct ExecutionStats{T<:AbstractFloat, V<:AbstractVector{T}, DF}
    status::Symbol
    elapsed_time::T
    iter::Int
    neval_jprod::Int
    neval_jtprod::Int
    primal_obj::T
    dual_obj::T
    solution::V
    residual::V
    optimality::T
    tracer::DF
end

function Base.show(io::IO, s::ExecutionStats)
    rel_res = norm(s.residual, Inf)
    @printf("\n")
    if s.status == :max_iter 
        @printf("Maximum number of iterations reached\n")
    elseif s.status == :optimal
        @printf("Optimality conditions satisfied\n")
    end
    @printf("Products with A   : %9d\n", s.neval_jprod)
    @printf("Products with A'  : %9d\n", s.neval_jtprod)
    @printf("Time elapsed (sec): %9.1f\n", s.elapsed_time)
    @printf("||Ax-b||₂         : %9.1e\n", rel_res)
    @printf("Optimality        : %9.1e\n", s.optimality)
end

"""
    value!(kl::KLLSModel, t; kwargs...)

Compute the dual objective of a KLLS model with respect to the scaling parameter `t`.
"""
function value!(kl::KLLSModel, t; kwargs...)
    @unpack λ, A = kl
    scale!(kl, t)
    s = solve!(kl; kwargs...)
    y = s.residual/λ
    dv = obj!(kl.lse, A'y) - log(t) - 1
    return dv
end

"""
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
    dv!(t) = value!(kl, t; atol=δ*atol, rtol=δ*rtol, logging=logging)
    t = Roots.find_zero(dv!, t; atol=atol, rtol=rtol, xatol=xatol, xrtol=xrtol, verbose=zverbose)
    return t, t*grad(kl.lse)
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