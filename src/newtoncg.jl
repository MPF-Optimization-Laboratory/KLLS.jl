import JSOSolvers: trunk

function dObj!(kl::KLLSModel{T,M,V}, y::V) where {T,M,V}
    @unpack A, b, λ, w, lse = kl 
    mul!(w, A', y)
    f = obj!(lse, w)
    return f + 0.5λ * y⋅y - b⋅y
end

function NLPModels.obj(kl::KLLSModel{T,M,V}, y::V) where {T,M,V}
    increment!(kl, :neval_jtprod)
    return dObj!(kl, y)
end

function dGrad!(kl::KLLSModel{T,M,V}, y::V, ∇f::V) where {T,M,V}
    @unpack A, b, λ, lse = kl
    p = grad(lse)
    @. ∇f = λ*y - b
    mul!(∇f, A, p, 1, 1)
    return ∇f
end

function NLPModels.grad!(kl::KLLSModel{T,M,V}, y::V, ∇f::V) where {T,M,V}
    increment!(kl, :neval_jprod)
    return dGrad!(kl, y, ∇f)
end

function dHess(kl::KLLSModel)
    @unpack A, λ, lse = kl
    H = hess(lse)
    ∇²dObj = A*H*A'
    if λ > 0
        ∇²dObj += λ*I
    end
    return ∇²dObj
end

function dHess_prod!(kl::KLLSModel{T, M, V}, y::V, v::V) where {T, M, V}
    @unpack A, λ, w, lse = kl
    # H = hess(lse)
    # v .= (A*(H*(A')*y)) + λ*y
    g = grad(lse)
    mul!(w, A', y)       # w =            A'y
    w .= g.*(w .- (g⋅w)) # w =  (G - gg')(A'y)
    mul!(v, A, w)        # v = A(G - gg')(A'y)
    if λ > 0
        v .+= λ*y
    end
    return v
end

function NLPModels.hprod!(kl::KLLSModel, y::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(eltype(y)))
    increment!(kl, :neval_jprod)
    increment!(kl, :neval_jtprod)
    return Hz = dHess_prod!(kl, z, Hz)
end

function solve!(kl::KLLSModel{T}; M=I, logging=0, monotone=true, max_time::Float64=30.0, kwargs...) where T
   
    # Reset counters
    reset!(kl)    

    # Tracer
    tracer = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[], Δ=Float64[], Δₐ_Δₚ=Float64[], cgits=Int[], cgmsg=String[])
    
    # Callback routine
    cb(nlp, solver, stats) = callback(
    kl, solver, M, stats, tracer, logging, max_time; kwargs...
    )
    
    # Call the Trunk solver
    if M === I
        trunk_stats = trunk(kl; callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time, monotone=monotone) 
    else
        trunk_stats = trunk(kl; M=M, callback=cb, atol=zero(T), rtol=zero(T)) 
    end
    
    stats = ExecutionStats(
        trunk_stats.status,
        trunk_stats.elapsed_time,
        trunk_stats.iter,
        neval_jprod(kl),
        neval_jtprod(kl),
        zero(T),
        trunk_stats.objective,
        grad(kl.lse),
        (kl.λ).*(trunk_stats.solution),
        trunk_stats.dual_feas,
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
"nonpositive curvature detected" => "neg curv",
"solution good enough given atol and rtol" => "✓",
"zero curvature detected" => "zer curv",
"maximum number of iterations exceeded" => "⤒",
"user-requested exit" => "user exit",
"time limit exceeded" => "time exit",
"unknown" => ""
)