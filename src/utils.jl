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

function histogram(stat::ExecutionStats; kwargs...)
    println("")
    UnicodePlots.histogram(stat.solution; kwargs...)
end

function value!(kl::KLLSModel{T}, t::T) where T
    @unpack λ, A = kl
    scale!(kl, t)
    s = solve!(kl)
    y = s.residual/λ
    v = s.dual_obj
    dv = obj!(kl.lse, A'y) - log(t) - 1
    return v, dv
end

"""
Maximize the dual objective of a KLLS model with respect to the scaling parameter `t`.
Returns the optimal primal solution.
"""
function maximize!(kl::KLLSModel{T}; t=one(T), kwargs...) where T
    dv!(t) = value!(kl, t)[2]
    t = Roots.find_zero(dv!, t; kwargs...)
    return t, t*grad(kl.lse)
end