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
    randDPModel(m, n; λ=1e-3) -> DPModel

Generate a random PT model. Arguments:
- `m`: number of rows of the matrix `A`
- `n`: number of columns of the matrix `A`
- `λ`: regularization parameter (default: 1e-3)
"""
function randDPModel(m, n; λ=1e-3)
    A = randn(m, n)
    xs = rand(n)
    xs ./= sum(xs)
    b = A * xs
    return DPModel(A, b, λ=λ)
end

"""
    histogram(s:ExecutionStats; kwargs...)

Plot a histogram of the solution.
"""
function histogram end

"""
    version() -> String

Return the current version of DualPerspective.jl package.
"""
function version()
    pkg_info = Pkg.project()
    for (uuid, info) in pkg_info.dependencies
        if info.name == Base.moduleroot(@__MODULE__).name
            return info.version
        end
    end
    # Fallback to the local project file if not installed as dependency
    return Pkg.TOML.parsefile(pkgdir(@__MODULE__), "Project.toml")["version"]
end
