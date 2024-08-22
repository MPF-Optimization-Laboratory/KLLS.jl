"""
Dual objective:

- base case (no scaling, unweighted 2-norm):
    f(y) = log∑exp(A'y) - 0.5λ y∙Cy - b∙y

- with scaling and weighted 2-norm:
    f(y) = τ log∑exp(A'y) - τ log τ + 0.5λ y∙Cy - b∙y
"""
function dObj!(kl::KLLSModel{T,M,CT,V}, y::V) where {T,M,CT,V}
    @unpack A, b, λ, C, w, lse, scale = kl 
    mul!(w, A', y)
    f = obj!(lse, w)
    return scale*f - scale*log(scale) + 0.5λ*dot(y, C, y) - b⋅y
end

function NLPModels.obj(kl::KLLSModel{T,M,CT,V}, y::V) where {T,M,CT,V}
    increment!(kl, :neval_jtprod)
    return dObj!(kl, y)
end

"""
Dual objective gradient

   ∇f(y) = τ A∇log∑exp(A'y) + λCy - b 

evaluated at `y`. Assumes that the objective was last evaluated at the same point `y`.
"""
function dGrad!(kl::KLLSModel, y::AbstractVector, ∇f::AbstractVector)
    @unpack A, b, λ, C, lse, scale = kl
    p = grad(lse)
    ∇f .= -b
    if λ > 0
        mul!(∇f, C, y, λ, 1)
    end
    mul!(∇f, A, p, scale, 1)
    return ∇f
end

function NLPModels.grad!(kl::KLLSModel{T,M,CT,V}, y::V, ∇f::V) where {T,M,CT,V}
    increment!(kl, :neval_jprod)
    return dGrad!(kl, y, ∇f)
end

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
Product of the dual objective Hessian with a vector `z`

    v ← ∇²d(y)z = τ A∇²log∑exp(A'y)Az + λCz,

where `y` is the point at which the objective was last evaluated.
"""
function dHess_prod!(kl::KLLSModel{T}, z, v) where T
    @unpack A, λ, C, w, lse, scale = kl
    g = grad(lse)
    mul!(w, A', z)                 # w =                  A'z
    w .= g.*(w .- (g⋅w))           # w =        (G - gg')(A'z)
    mul!(v, A, w, scale, zero(T))  # v = scale*A(G - gg')(A'z)
    if λ > 0
        mul!(v, C, z, λ, 1)        # v += λCz
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
        trunk_stats.elapsed_time,       # elapsed time
        trunk_stats.iter,               # number of iterations
        neval_jprod(kl),                # number of products with A
        neval_jtprod(kl),               # number of products with A'
        zero(T),                        # primal objective TODO
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