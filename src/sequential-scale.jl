"""
    value!(kl::KLLSModel, t; kwargs...)

Compute the dual objective of a KLLS model with respect to the scaling parameter `t`.
"""
function value!(kl::KLLSModel, t; jprods=Int[0], jtprods=Int[0], kwargs...)
    @unpack λ, A = kl
    scale!(kl, t)
    s = solve!(kl; kwargs...)
    y = s.residual/λ
    dv = obj!(kl.lse, A'y) - log(t) - 1
    jprods[1] += neval_jprod(kl)
    jtprods[1] += neval_jtprod(kl)
    update_y0!(kl, s.residual ./ kl.λ) # Set the next runs starting point to the radial projection
    return dv
end

struct SequentialSolve end

"""
    solve!(ss::SSModel; kwargs...) -> t, xopt, jprods

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
function solve!(
    kl::KLLSModel{T},
    ::SequentialSolve;
    t=one(T),
    rtol=1e-6,
    atol=1e-6,
    xatol=1e-6,
    xrtol=1e-6,
    δ=1e-2,
    zverbose=false,
    logging=0,
    kwargs...
    ) where T

    ss = SSModel(kl)

    # Initalize counter for mat-vec products
    jprods = Int[0]
    jtprods = Int[0]
    
    # Setup trackers
    tracker = Roots.Tracks()
    tracer = DataFrame(iter=Int[], scale=T[], vpt=T[], norm∇d=T[], cgits=Int[], cgmsg=String[])

    # Find optimal t
    start_time = time()
    dv!(t) = value!(ss.kl, t; jprods=jprods, jtprods=jtprods, atol=δ*atol, rtol=δ*rtol, logging=logging)
    t = Roots.find_zero(dv!, t; tracks=tracker, atol=atol, rtol=rtol, xatol=xatol, xrtol=xrtol, verbose=zverbose)
    elapsed_time = time() - start_time

    # Solve one final time
    scale!(ss.kl, t)
    final_run_stats = solve!(ss.kl, atol=δ*atol, rtol=δ*rtol, logging=logging, reset_counters=false)

    status = :unknown
    if tracker.convergence_flag == :x_converged
        status = :optimal
    end

    stats = ExecutionStats(
        status,
        elapsed_time,                   # elapsed time
        tracker.steps,                  # number of iterations
        jprods[1],                      # number of products with A
        jtprods[1],                     # number of products with A'
        zero(T),                        # TODO: primal objective
        final_run_stats.dual_obj,       # dual objective
        final_run_stats.solution,       # primal solution `x`
        final_run_stats.residual,       # residual r = λy
        final_run_stats.optimality,     # norm of gradient of the dual objective
        tracer                          # TODO: tracer to store iteration info 
    ) 

    return stats
end
