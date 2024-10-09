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
    m = ss.kl.meta.nvar

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
                    linesearch = RobustNonMonotoneLineSearch(),
                    linsolve = KrylovJL_MINRES(verbose=0, itmax=50),
                )
            )
    
    for iter in 1:100
        @suppress_err step!(nlcache)

        y = @view nlcache.u[1:m]
        t =       nlcache.u[end]  
        ∇d = @view nlcache.fu[1:m]
        vpt =      nlcache.fu[end]
       
        norm∇d = norm(∇d, Inf)
        log_items = (iter, t, vpt, norm∇d, nlcache.descent_cache.lincache.lincache.cacheval.stats.niter, nlcache.descent_cache.lincache.lincache.cacheval.stats.status) 
        trace && push!(tracer, log_items)
        
        if nlcache.retcode ≠ ReturnCode.Default
            break
        end
    end
    
    return nlcache, tracer
end