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
