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
