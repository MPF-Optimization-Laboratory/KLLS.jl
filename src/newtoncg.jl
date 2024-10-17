"""
Compute f(y) = logΣexp(A'y - c) and it's gradient,
stored in the `lse` internal buffer.
"""
function lseatyc!(kl, y)
    @unpack A, c, nbuf, lse = kl
    nbuf .= c
    mul!(nbuf, A', y, 1, -1)
    return obj!(lse, nbuf)
end

"""
Dual objective:

- base case (no scaling, unweighted 2-norm):
    f(y) = log∑exp(A'y - c) - 0.5λ y∙Cy - b∙y

- with scaling and weighted 2-norm:
    f(y) = τ log∑exp(A'y - c) - τ log τ + 0.5λ y∙Cy - b∙y
"""
function dObj!(kl::KLLSModel, y)
    @unpack b, λ, C, scale = kl 
    increment!(kl, :neval_jtprod)
    f = lseatyc!(kl, y)
    return scale*f - scale*log(scale) + 0.5λ*dot(y, C, y) - b⋅y
end

NLPModels.obj(kl::KLLSModel, y) = dObj!(kl, y)

"""
Dual objective gradient

   ∇f(y) = τ A∇log∑exp(A'y-c) + λCy - b 

evaluated at `y`. Assumes that the objective was last evaluated at the same point `y`.
"""
function dGrad!(kl::KLLSModel, y, ∇f)
    @unpack A, b, λ, C, lse, scale = kl
    increment!(kl, :neval_jprod)
    p = grad(lse)
    ∇f .= -b
    if λ > 0
        mul!(∇f, C, y, λ, 1)
    end
    mul!(∇f, A, p, scale, 1)
    return ∇f
end

NLPModels.grad!(kl::KLLSModel, y, ∇f) = dGrad!(kl, y, ∇f)

function dHess(kl::KLLSModel)
    @unpack A, λ, C, lse, scale = kl
    H = hess(lse)
    ∇²dObj = scale*(A*H*A')
    if λ > 0
        ∇²dObj += λ*C
    end
    return ∇²dObj
end

"""
    dHess_prod!(kl::KLLSModel{T}, z, Hz) where T

Product of the dual objective Hessian with a vector `z`

    Hz ← ∇²d(y)z = τ A∇²log∑exp(A'y)Az + λCz,

where `y` is the point at which the objective was last evaluated.
"""
function dHess_prod!(kl::KLLSModel, z, Hz)
    @unpack A, λ, C, nbuf, lse, scale = kl
    w = nbuf
    increment!(kl, :neval_jprod)
    increment!(kl, :neval_jtprod)
    g = grad(lse)
    mul!(w, A', z)                 # w =                  A'z
    w .= g.*(w .- (g⋅w))           # w =        (G - gg')(A'z)
    mul!(Hz, A, w, scale, 0)       # v = scale*A(G - gg')(A'z)
    if λ > 0
        mul!(Hz, C, z, λ, 1)       # v += λCz
    end
    return Hz
end

function NLPModels.hprod!(kl::KLLSModel{T}, ::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(T)) where T
    return Hz = dHess_prod!(kl, z, Hz)
end

function solve!(
    kl::KLLSModel{T};
    M=I,
    logging=0,
    monotone=true,
    max_time::Float64=30.0,
    kwargs...) where T
   
    # Reset counters
    reset!(kl)    

    # Tracer
    tracer = DataFrame(iter=Int[], dual_obj=T[], r=T[], Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])
    
    # Callback routine
    cb(kl, solver, stats) =
        callback(kl, solver, M, stats, tracer, logging, max_time; kwargs...)
    
    # Call the Trunk solver
    if M === I
        trunk_stats = trunk(kl; callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time, monotone=monotone) 
    else
        trunk_stats = trunk(kl; M=M, callback=cb, atol=zero(T), rtol=zero(T)) 
    end
    
    stats = ExecutionStats(
        trunk_stats.status,
        trunk_stats.elapsed_time,       # elapsed time
        trunk_stats.iter,               # number of iterations
        neval_jprod(kl),                # number of products with A
        neval_jtprod(kl),               # number of products with A'
        zero(T),                        # TODO: primal objective
        trunk_stats.objective,          # dual objective
        (kl.scale).*grad(kl.lse),       # primal solultion `x`
        (kl.λ).*(trunk_stats.solution), # residual r = λy
        trunk_stats.dual_feas,          # norm of the gradient of the dual objective
        tracer
    )
end
const newtoncg = solve!

function callback(
    kl::KLLSModel{T},
    solver,
    M,
    trunk_stats,
    tracer,
    logging,
    max_time;
    atol::T = DEFAULT_PRECISION(T),
    rtol::T = DEFAULT_PRECISION(T),
    max_iter::Int = typemax(Int),
    trace::Bool = false,
    ) where T
    
    dObj = trunk_stats.objective 
    iter = trunk_stats.iter
    r = trunk_stats.dual_feas # = ||∇ dual obj(x)||
    # r = norm(solver.gx)
    Δ = solver.tr.radius
    actual_to_predicted = solver.tr.ratio
    cgits = solver.subsolver.stats.niter
    cgexit = cg_msg[solver.subsolver.stats.status]
    ε = atol + rtol * kl.bNrm
    
    # Test exit conditions
    tired = iter >= max_iter
    optimal = r < ε 
    done = tired || optimal
    
    log_items = (iter, dObj, r, Δ, actual_to_predicted, cgits, cgexit) 
    trace && push!(tracer, log_items)
    if logging > 0 && iter == 0
        println("\n", kl)
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
    if trunk_stats.status == :unkown
        return
    end
    
    # Update the preconditioner
    update!(M)
end

const cg_msg = Dict(
"on trust-region boundary" => "⊕",
"found approximate minimum least-squares solution" => "min soln",
"nonpositive curvature detected" => "neg curv",
"solution good enough given atol and rtol" => "✓",
"zero curvature detected" => "zer curv",
"maximum number of iterations exceeded" => "⤒",
"found approximate zero-residual solution" => "zero res",
"user-requested exit" => "user exit",
"time limit exceeded" => "time exit",
"unknown" => ""
)

#######################################################################
# Nonlinear Least Squares via NonlinearSolve.jl
#######################################################################

"""
    Fx = residual!(kl, yt, Fx)

Compute the residual of the gradient of the dual problem 

    F(y, t) = [ A(tx(y)) + λCy - b ]

""" 
function residual!(kl::KLLSModel, y, Fx)
	# increment!(kl, :neval_residual)
    lseatyc!(kl, y)                 # Calculate gradient
	Fx = dGrad!(kl, y, Fx)          # r ≡ Fx = ∇d(y)	
	return Fx
end

"""
    Jy = jprod_residual!(kl, y, z, Jy)

Compute the Jacobian-vector product, 

    ∇²d(A'y)z := Jy
"""
function jprod_residual!(kl::KLLSModel, z, Jy)
    # increment!(kl, :neval_jprod_residual)
    dHess_prod!(kl, z, Jy)
    return Jy
end

function nlresidual!(F, y, kl::KLLSModel)
    residual!(kl, y, F)
end

function nljprod!(Jy, z, y, kl::KLLSModel)
    jprod_residual!(kl, z, Jy)
end

struct NewtonEQKLLS end

function solve!(
    kl::KLLSModel{T},
    ::NewtonEQKLLS;
    y0 = begin
        m = kl.meta.nvar
        zeros(T, m)
    end,
    atol = DEFAULT_PRECISION(T),
    rtol = DEFAULT_PRECISION(T),
    max_time=30.0,
    max_iter=1000,
    trace::Bool = false,
    kwargs...) where T
    
    reset!(kl) # reset counters
    m = kl.meta.nvar

    tracer = DataFrame(iter=Int[], norm∇d=T[], cgits=Int[], cgmsg=String[])
   
    # Setup the NonlinearSolve objects
    nlf = NonlinearFunction(nlresidual!, jvp=nljprod!)
    prob = NonlinearProblem(nlf, y0, kl)
    nlcache = init(
                prob,
                reltol=rtol,
                abstol=atol,
                show_trace = Val(false),
                store_trace = Val(false),
                NewtonRaphson(
                    linesearch = RobustNonMonotoneLineSearch(),
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

        ∇d = nlcache.fu
       
        norm∇d = norm(∇d, Inf)
        log_items = (iter, norm∇d, nlcache.descent_cache.lincache.lincache.cacheval.stats.niter, nlcache.descent_cache.lincache.lincache.cacheval.stats.status) 
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
    y = nlcache.u
    x = kl.scale.*grad(kl.lse)
    ∇d = nlcache.fu
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
