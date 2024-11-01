"""
Solve the LP problem:

    min ⟨c, x⟩
    subject to Ax = b, x ≥ 0

using the self-scaled solver by solving the regularized problem:

    minimize_{x ∈ ℝ₊ⁿ} (1/(2λ)) * ||Ax - b||² + ⟨c, x⟩ + ε * H(x)

which is equivalent to:

    minimize_{x ∈ ℝ₊ⁿ} (1/(2λε)) * ||Ax - b||² + ⟨c/ε, x⟩ + H(x)

where ε is the relaxation constant, λ is the feasibility constant, and H(x) is the entropy function.
"""
struct LPModel{T, SS<:SSModel{T}}
    ss::SS   # Self-scaled model
    ε::T     # Relaxation constant
    λ::T     # Feasibility constant
end

"""
Constructs an `LPModel` for the LP problem.

# Arguments
- `A`: Constraint matrix
- `b`: Right-hand side vector
- `c`: Cost vector
- `ε`: Relaxation constant
- `λ`: Feasibility constant
- `maximize`: If `true`, solves max ⟨c, x⟩
- `kwargs`: Additional keyword arguments

# Returns
An instance of `LPModel`.
"""
function LPModel(
    A,
    b,
    c;
    ε=1e-3,
    λ=1e-3,
    maximize=false,
    kwargs...
)
    c = c / ε
    if maximize
        c = -c
    end
    kl = KLLSModel(A=A, b=b, c=c, λ=λ, kwargs...)
    regularize!(kl, ε * λ)
    ss = SSModel(kl)
    return LPModel(ss, ε, λ)
end

"""
Custom display for `LPModel` instances.
"""
function Base.show(io::IO, lp::LPModel)
    println(io, "KL regularized LP" *
                (lp.ss.kl.name == "" ? "" : ": " * lp.ss.kl.name))
    println(io, @sprintf("   m = %10d  bNrm = %7.1e", size(lp.ss.kl.A, 1), lp.ss.kl.bNrm))
    println(io, @sprintf("   n = %10d  λ    = %7.1e", size(lp.ss.kl.A, 2), lp.λ))
    println(io, @sprintf("       %10s  ε    = %7.1e", " ", lp.ε))
end

"""
Adjusts the feasibility constant `λ` in the `LPModel`.

# Arguments
- `lp`: The `LPModel` instance.
- `λ`: New feasibility constant.
"""
function regularize_feasability!(lp::LPModel, λ)
    regularize!(lp.ss.kl, λ * lp.ε)
    lp.λ = λ
end

"""
Adjusts the relaxation constant `ε` in the `LPModel`.

# Arguments
- `lp`: The `LPModel` instance.
- `ε`: New relaxation constant.
"""
function regularize_relaxation!(lp::LPModel, ε)
    regularize!(lp.ss.kl, lp.λ * ε)
    lp.ss.kl.c = lp.ss.kl.c / ε
    lp.ε = ε
end

"""
Resets the `LPModel` to its initial state.
"""
NLPModels.reset!(lp::LPModel) = NLPModels.reset!(lp.ss)

"""
Solves the LP problem using the self-scaled solver.

# Arguments
- `lp`: The `LPModel` instance.
- `logging`: Level of logging detail (default: `0`).
- `monotone`: Enforce monotonicity (default: `true`).
- `max_time`: Maximum solving time in seconds (default: `30.0`).
- `kwargs`: Additional keyword arguments.

# Returns
A statistics object containing the solution and status.

# Notes
- If the solution is found but the residual norm exceeds `1e-1`, the status is set to `:infeasible`.
"""
function solve!(
    lp::LPModel{T};
    logging=0,
    monotone=true,
    max_time=30.0,
    kwargs...
) where T
    klls_stats = solve!(
        lp.ss,
        TrunkLS(),
        logging=logging,
        monotone=monotone,
        max_time=max_time,
        kwargs...
    )

    if klls_stats.status == :optimal && norm(klls_stats.residual) > 1e-1
        klls_stats.status = :infeasible
    end

    return klls_stats
end
