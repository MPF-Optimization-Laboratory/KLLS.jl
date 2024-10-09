function solve!(cls::CLSModel{T}; logging=0, monotone=true, max_time::Float64=30.0, kwargs...) where T
   
    # Reset counters
    reset!(cls)    

    # Tracer
    tracer = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[], Δ=Float64[], Δₐ_Δₚ=Float64[], cgits=Int[], cgmsg=String[])
    
    # Callback routine
    cb(nlp, solver, stats) = callback(
    cls, solver, M, stats, tracer, logging, max_time; kwargs...
    )
    
    # Call the Trunk solver
    trunk_stats = trunk(cls; callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time, monotone=monotone) 
    
    stats = ExecutionStats(
        trunk_stats.status,
        trunk_stats.elapsed_time,           # elapsed time
        trunk_stats.iter,                   # number of iterations
        neval_jprod(cls),                   # number of products with A
        neval_jtprod(cls),                  # number of products with A'
        zero(T),                            # TODO: primal objective
        trunk_stats.objective,              # dual objective
        (cls.scale).*grad(cls.reg),         # primal solultion `x`
        (cls.λ).*(trunk_stats.solution),    # residual r = λy
        trunk_stats.dual_feas,              # norm of the gradient of the dual objective
        tracer
    )
end

function callback(
    cls::CLSModel{T},
    solver,
    M,
    trunk_stats,
    tracer,
    logging,
    max_time;
    atol::T = √eps(T),
    rtol::T = √eps(T),
    max_iter::Int = typemax(Int),
    trace::Bool = false,
    ) where T
    
    dObj = trunk_stats.objective 
    iter = trunk_stats.iter
    r = trunk_stats.dual_feas # = ||∇ dual obj(x)|| = ||λy||
    # r = norm(solver.gx)
    Δ = solver.tr.radius
    actual_to_predicted = solver.tr.ratio
    cgits = solver.subsolver.stats.niter
    cgexit = cg_msg[solver.subsolver.stats.status]
    ε = atol + rtol * cls.bNrm
    
    # Test exit conditions
    tired = iter >= max_iter
    optimal = r < ε 
    done = tired || optimal
    
    log_items = (iter, dObj, r, Δ, actual_to_predicted, cgits, cgexit) 
    trace && push!(tracer, log_items)
    if logging > 0 && iter == 0
        println("\n", cls)
        println("Solver parameters:")
        @printf("   atol = %7.1e  max time (sec) = %7d\n", atol, max_time)
        @printf("   rtol = %7.1e  target ∥r∥<ε   = %7.1e\n\n", rtol, ε)
        @printf("%7s  %9s  %9s  %9s  %9s  %6s  %10s\n",
        "iter","dual Obj","∥∇dObj∥","Δ","Δₐ/Δₚ","cg its","cg msg")
    end
    if logging > 0 && (mod(iter, logging) == 0 || done)
        @printf("%7d  %9.2e  %9.2e  %9.1e %9.1e  %6d   %10s\n", (log_items...))
    end
    
    if optimal
        trunk_stats.status = :optimal
    elseif tired
        trunk_stats.status = :max_iter
    end
    if trunk_stats.status == :unknown
        return
    end
    
    # Update the preconditioner
    update!(M)
end

const cg_msg = Dict(
"on trust-region boundary" => "⊕",
"nonpositive curvature detected" => "neg curv",
"solution good enough given atol and rtol" => "✓",
"zero curvature detected" => "zer curv",
"maximum number of iterations exceeded" => "⤒",
"user-requested exit" => "user exit",
"time limit exceeded" => "time exit",
"unknown" => ""
)