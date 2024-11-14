struct SSModel{T, S, K<:KLLSModel{T}} <: AbstractNLSModel{T,S}
    kl::K
    meta::NLPModelMeta{T,S}
    nls_meta::NLSMeta{T,S}
    counters::NLSCounters
end

"""
    SSModel(kl::KLLSModel) -> SSModel

This model is a container for `kl` and the augmented problem for the self-scaled model in the variables (y,t).

Default starting point is `(kl.x0, 1.0)`
"""
function SSModel(kl::KLLSModel{T}) where T
    m = kl.meta.nvar
    y0 = kl.meta.x0
    meta = NLPModelMeta(
        m+1,
        x0 = vcat(y0, one(T)),
        name = "Scaled Simplex Model"
    )
    nls_meta = NLSMeta{T, Vector{T}}(m+1, m+1)
    return SSModel(kl, meta, nls_meta, NLSCounters())
end

function Base.show(io::IO, ss::SSModel)
    println(io, "Self-scaled model")
    show(io, ss.kl)
end

NLPModels.reset!(ss::SSModel) = NLPModels.reset!(ss.kl)

"""
    residual!(ss, yt, Fx)

Compute the residual in the self-scaling optimality conditions augmented problem, which concatenate the dual residual with the optimal scaling condition:

    F(y, t) = [ ∇d(y)
                logexp(A'y) - log(t) - 1 ]

where

    ∇d(y) = A(tx(y)) + λCy - b
    x(y) = ∇logexp(A'y)
""" 
function NLPModels.residual!(ss::SSModel, yt, Fx)
	increment!(ss, :neval_residual)
    kl = ss.kl
	@unpack A, c, lse = kl
	m = kl.meta.nvar
    
    r = @view Fx[1:m]
	y = @view yt[1:m]
	t =       yt[end]
    
    scale!(kl, t)             # Apply the latest scaling factor
    f = lseatyc!(kl, y)       # f = logΣexp(A'y). Needed before grad eval
	dGrad!(kl, y, r)          # r ≡ Fx[1:m] = ∇d(y)	
	Fx[end] = f - log(t) - 1  # optimal scaling condition
	return Fx
end

"""
    Jyt = jprod_residual!(ss, yt, zα, Jyt)

Compute the Jacobian-vector product, 

    (1) [ ∇²d(A'y)  Ax  ][ z ] := [ Jy ]  where x:=x(y)
    (2) [ (Ax)'     -1/t][ α ] := [ Jt ]
"""
function NLPModels.jprod_residual!(ss::SSModel, yt, zα, Jyt)

    kl = ss.kl
    @unpack A, lse, mbuf = kl
    Ax = mbuf
    m = kl.meta.nvar
    x = grad(lse)

    increment!(ss, :neval_jprod_residual)

    Jy = @view Jyt[1:m]
    t = yt[end]
    z = @view zα[1:m]
    α = zα[end]
   
    mul!(Ax, A, x)

    # Equation (1)
    dHess_prod!(kl, z, Jy)  # Jy = ∇²d(A'y)z
    Jy .+= α*Ax             # Jy += αAx
    
    # Equation (2)  
    Jyt[end] = z⋅Ax - α/t

    return Jyt
end

function NLPModels.jtprod_residual!(ss::SSModel, yt, wα, Jyt)
    increment!(ss, :neval_jtprod_residual)
    NLPModels.jprod_residual!(ss, yt, wα, Jyt)
end

#######################################################################
# Nonlinear Least Squares via TrunkLS
#######################################################################

struct TrunkLS end

solve!(ss::SSModel; kwargs...) = solve!(ss, TrunkLS(); kwargs...)

"""
    solve!(ss::SSModel, ::TrunkLS; kwargs...)

Solve the self-scaled model using Gauss-Newton, via the TrunkLS algorithm.
"""
function solve!(
    ss::SSModel{T},
    ::TrunkLS;
    logging=0,
    monotone=true,
    max_time=30.0,
    kwargs...) where T

    reset!(ss) # reset counters

    tracer = DataFrame(iter=Int[], scale=T[], vpt=T[], dual_grad=T[], r=T[], Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])

    # Callback routine
    cb(ss::SSModel, solver, stats) =
      callback(ss, solver, stats, tracer, logging, max_time; kwargs...)
    
    trunk_stats =
      trunk(ss; callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time, monotone=monotone) 

    # Optimality. Report the maximum ∇d(y) and v'(t)
    optimality = sqrt(obj(ss, trunk_stats.solution))

    kl = ss.kl
    x = kl.scale.*grad(kl.lse)
    y = @view trunk_stats.solution[1:end-1]
    stats = ExecutionStats(
        trunk_stats.status,          # status
        trunk_stats.elapsed_time,    # elapsed time
        trunk_stats.iter,            # iterations
        neval_jprod(kl),             # count products with A
        neval_jtprod(kl),            # count products with A'
        zero(T),                     # TODO: primal objective
        trunk_stats.objective,       # dual objective
        x,                           # primal solultion `x`
        (kl.λ)*y,                    # residual r = λy
        optimality,       # norm of the gradient of the dual objective
        tracer                       # TODO: tracer 
    )
    return stats
end

function callback(
    ss::SSModel{T},
    solver,
    trunk_stats,
    tracer,
    logging,
    max_time;
    atol = DEFAULT_PRECISION(T),
    rtol = DEFAULT_PRECISION(T),
    max_iter::Int = typemax(Int),
    trace::Bool = false,
    ) where T
  
    
    # Norm of the (unscaled) dual gradient ∇d(y)
    m = length(ss.kl.b)
    ∇d = @view solver.Fx[1:m]
    norm∇d = norm(∇d, Inf)
    vpt = solver.Fx[end]

    iter = trunk_stats.iter # = number of iterations
    r = trunk_stats.dual_feas # = ||∇F⋅F(y,t)||
    scale = solver.x[end]
    Δ = solver.tr.radius 
    actual_to_predicted = solver.tr.ratio
    cgits = solver.subsolver.stats.niter
    cgexit = cg_msg[solver.subsolver.stats.status]
    ε = atol + rtol * ss.kl.bNrm
    
    # Test exit conditions
    tired = iter >= max_iter
    optimal = r < ε 
    done = tired || optimal
    
    log_items = (iter, scale, vpt, norm∇d, r, Δ, actual_to_predicted, cgits, cgexit) 
    trace && push!(tracer, log_items)
    if logging > 0 && iter == 0
        println("\n", ss)
        println("Solver parameters:")
        @printf("   atol = %7.1e  max time (sec) = %7d\n", atol, max_time)
        @printf("   rtol = %7.1e  target ∥r∥<ε   = %7.1e\n\n", rtol, ε)
        @printf("%7s  %8s  %9s  %8s  %8s  %7s  %7s  %4s  %10s\n",
        "iter","t","v'(t)","∥∇d(y)∥","∥∇F⋅F∥","Δ","Δₐ/Δₚ","cg","cg msg")
    end
    if logging > 0 && (mod(iter, logging) == 0 || done)
        @printf("%7d  %8.2e  %9.2e  %8.2e  %8.2e  %7.1e  %7.1e  %4d  %10s\n", (log_items...))
    end
    
    if optimal
        trunk_stats.status = :optimal
    elseif tired
        trunk_stats.status = :max_iter
    end
    if trunk_stats.status == :unkown
        return
    end
end

#######################################################################
# Nonlinear Least Squares via NonlinearSolve.jl
#######################################################################

# These methods just rearrange the inputs to the NonlinearSolve.jl API
function nlresidual!(F, yt, ss::SSModel)
    residual!(ss, yt, F)
end

function nljprod!(Jyt, zα, yt, ss::SSModel)
    jprod_residual!(ss, yt, zα, Jyt)
end

struct NewtonEQ end

function solve!(
    ss::SSModel{T},
    ::NewtonEQ;
    y0 = begin
        m = ss.kl.meta.nvar
        zeros(T, m)
    end,
    t0 = one(T),
    atol = DEFAULT_PRECISION(T),
    rtol = DEFAULT_PRECISION(T),
    logging=0,
    max_time=30.0,
    max_iter=1000,
    trace::Bool = false,
    kwargs...) where T
    
    reset!(ss) # reset counters
    kl = ss.kl
    m = kl.meta.nvar

    tracer = DataFrame(iter=Int[], scale=T[], vpt=T[], norm∇d=T[], cgits=Int[], cgmsg=String[])
   
    # Setup the NonlinearSolve objects
    nlf = NonlinearFunction(nlresidual!, jvp=nljprod!)
    prob = NonlinearProblem(nlf, vcat(y0, t0), ss)
    nlcache = init(
                prob,
                reltol=rtol,
                abstol=atol,
                show_trace = Val(false),
                store_trace = Val(false),
                NewtonRaphson(
                    # linesearch = RobustNonMonotoneLineSearch(),
                    linesearch = BackTracking(),
                    linsolve = KrylovJL_MINRES(verbose=0, itmax=50),
                )
            )
   
    start_time = time()
    elapsed_time = 0.0
    iter = 0
    while true 
        
        # Break if time is up
        elapsed_time = time() - start_time
        iter += 1
        elapsed_time > max_time && break
        iter > max_iter && break

        # Take a Newton step
        step!(nlcache)

        t = nlcache.u[end]  
        ∇d = @view nlcache.fu[1:m]
        vpt = nlcache.fu[end]
       
        norm∇d = norm(∇d, Inf)
        log_items = (iter, t, vpt, norm∇d, nlcache.descent_cache.lincache.lincache.cacheval.stats.niter, nlcache.descent_cache.lincache.lincache.cacheval.stats.status) 
        trace && push!(tracer, log_items)
        
        if nlcache.retcode ≠ ReturnCode.Default
            break
        end
    end

    # Test nlcache.retcode to return one of the following symbols:
    # :optimal, :max_iter, :unknown
    status = if nlcache.retcode == ReturnCode.Success
        :optimal
    elseif nlcache.retcode == ReturnCode.MaxIters
        :max_iter
    else
        :unknown
    end

    λ = kl.λ
    y = @view nlcache.u[1:m]
    x = kl.scale.*grad(kl.lse)
    ∇d = @view nlcache.fu[1:m]
    r = λ*y
    stats = ExecutionStats(
        status,
        elapsed_time,       # elapsed time
        iter,               # number of iterations
        neval_jprod(kl),    # number of products with A
        neval_jtprod(kl),   # number of products with A'
        zero(T),            # TODO: primal objective
        zero(T),            # dual objective
        x,                  # primal solultion `x`
        r,                  # residual r = λy
        norm(∇d),           # norm of gradient of the dual objective
        tracer              # TODO: tracer 
    ) 
    
    return stats
end

