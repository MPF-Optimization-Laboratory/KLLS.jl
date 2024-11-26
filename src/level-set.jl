struct LevelSet end

function solve!(
    kl::KLLSModel{T},
    ::LevelSet;
    α::T=1.5,
    σ=kl.bNrm^2 / (2 * kl.λ),
    t=1,
    logging=0,
    atol=1e-3,
    rtol=1e-3,
    max_time::Float64=30.0,
    kwargs...
) where {T}
    @assert 1 < α < 2 "α must be in the open interval (1, 2)"
    @assert t > 0 "t0 must be positive"
    @assert σ >= 0 "σ must be non-negative"

    scale!(kl, t)

    it = 0
    tracer = DataFrame(iter=Int[], l=T[], u=T[], u_over_l=T[], s=T[])
    l, u, s = 0.0, 0.0, 0.0
    solver = TrunkSolver(kl)
    start_time = time()


    while true
        it += 1
        l, u, s = oracle!(kl, α, σ, solver, tracer, logging=logging, max_time=max_time) # TODO: weird max time
        tk = t - l / s
        
        @assert tk > 0 "New scale must be positive"

        small_step = abs(tk - t) ≤ atol + t*rtol
        min_value = u ≤ atol + σ*rtol
        done = small_step || min_value

        if logging > 0
            @printf("lvl itn: %7d ℓ: %9.2e u: %9.2e s: %9.2e tₖ: %9.2e  Δₜ: %9.2e\n", it, l, u, s, tk, abs(tk - t))
            if done && small_step
                println("Stopping due to small step in t")
            elseif done
                println("Stopping due to small upper bound")
            end
        end

        if done
            break
        end
        t = tk
        scale!(kl, t)
    end

    final_soln = solve!(kl, logging=logging, reset_counters=false, kwargs...)

    runtime = time() - start_time

    stats = ExecutionStats(
        final_soln.status,
        runtime,                        # elapsed time
        it,                             # number of iterations
        neval_jprod(kl),                # number of products with A
        neval_jtprod(kl),               # number of products with A'
        final_soln.primal_obj,          # primal objective
        final_soln.dual_obj,            # dual objective
        final_soln.solution,            # primal solution `x`
        final_soln.residual,            # residual r = λy
        final_soln.optimality,          # norm of the gradient of the dual objective
        tracer
    )
    return stats
end

function oracle!(
    kl::KLLSModel{T},
    α::T,
    σ::T,
    solver::TrunkSolver,
    tracer::DataFrame;
    logging=0,
    max_time::Float64=30.0,
    kwargs...
) where {T}
    # return values (l, u, s)
    ret = [0.0, 0.0, 0.0]

    # Reset the solver
    SolverCore.reset!(solver, kl)

    # Callback routine
    cb(kl, solver, stats) =
        oracle_callback(kl, solver, stats, tracer, logging, α, σ, ret; kwargs...)

    stats = SolverCore.solve!(solver, kl; x=kl.meta.x0, callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time)

    return ret
end

function oracle_callback(
    kl::KLLSModel{T},
    solver,
    trunk_stats,
    tracer,
    logging,
    α,
    σ,
    ret;
    atol::T=DEFAULT_PRECISION(T),
    rtol::T=DEFAULT_PRECISION(T),
    max_iter::Int=typemax(Int),
    trace::Bool=false,
) where {T}
    y = solver.x
    x = kl.scale * grad(kl.lse)
    dObj = -trunk_stats.objective - σ
    iter = trunk_stats.iter
    r = trunk_stats.dual_feas # = ||∇ dual obj(x)||
    Δ = solver.tr.radius
    actual_to_predicted = solver.tr.ratio
    cgits = solver.subsolver.stats.niter
    cgexit = get(cg_msg, solver.subsolver.stats.status, "default")
    ε = atol + rtol * kl.bNrm
    pObj = pObj!(kl::KLLSModel, x) - σ

    # Test exit conditions
    tired = iter >= max_iter
    optimal = r < ε
    done = tired || optimal

    # Logging & Tracing
    log_items = (iter, dObj, pObj, pObj / dObj, r, Δ, actual_to_predicted, cgits, cgexit)
    trace && push!(tracer, log_items)
    if logging > 0 && iter == 0
        println("Inside loop:")
        @printf("%7s  %9s  %9s  %9s  %9s  %9s  %9s  %6s  %10s\n",
            "iter", "dObj-σ", "pObj-σ", "ratio", "∥∇dObj∥", "Δ", "Δₐ/Δₚ", "cg its", "cg msg")
    end
    if logging > 0 && (mod(iter, logging) == 0 || done)
        @printf("%7d  %9.2e  %9.2e %9.1f %9.1e %9.1e %9.1e  %6d   %10s\n", (log_items...))
    end

    if optimal
        trunk_stats.status = :optimal
    elseif tired
        trunk_stats.status = :max_iter
    elseif pObj < α * dObj && dObj > 0
        st = -obj!(kl.lse, kl.A'y) + log(kl.scale) + 1
        ret .= [dObj, pObj, st]
        trunk_stats.status = :user # Ends the oracle iterations
        update_y0!(kl, y)          # Set up starting point for next iteration
    end
end
