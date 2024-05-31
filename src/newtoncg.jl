import NLPModels: NLPModels, NLPModelMeta, AbstractNLPModel, Counters, increment!, neval_jprod, neval_jtprod
import JSOSolvers: trunk

struct KLLSModel{T, M, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    data::KLLSData{T, M, S}
end

function KLLSModel(data)
    m = size(data.A, 1)
    KLLSModel(NLPModelMeta(m, name="KLLS"), Counters(), data)
end

function NLPModels.obj(nlp::KLLSModel, y::AbstractVector)
    increment!(nlp, :neval_jtprod)
    return dObj!(nlp.data, y)
end

function NLPModels.grad!(nlp::KLLSModel, y::AbstractVector, ∇f::AbstractVector)
    increment!(nlp, :neval_jprod)
    return dGrad!(nlp.data, y, ∇f)
end

function NLPModels.hprod!(nlp::KLLSModel, y::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(eltype(y)))
    increment!(nlp, :neval_jprod)
    increment!(nlp, :neval_jtprod)
    return Hz = dHess_prod!(nlp.data, z, Hz)
end

function solve!(data::KLLSData{T}; M=I, logging=0, monotone=true, max_time::Float64=30.0, kwargs...) where T
    
    # Build the NLP model from the KL data
    nlp = KLLSModel(data)
    
    # Tracer
    tracer = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[], Δ=Float64[], Δₐ_Δₚ=Float64[], cgits=Int[], cgmsg=String[])
    
    # Callback routine
    cb(nlp, solver, stats) = callback(
        nlp, solver, M, stats, tracer, logging; kwargs...
        )
    
    # Call the Trunk solver
    if M === I
        trunk_stats = trunk(nlp; callback=cb, atol=zero(T), rtol=zero(T), max_time=max_time, monotone=monotone) 
    else
        trunk_stats = trunk(nlp; M=M, callback=cb, atol=zero(T), rtol=zero(T)) 
    end
    
    stats = ExecutionStats(
        trunk_stats.status,
        trunk_stats.elapsed_time,
        trunk_stats.iter,
        neval_jprod(nlp),
        neval_jtprod(nlp),
        zero(T),
        trunk_stats.objective,
        grad(data.lse),
        (data.λ).*(trunk_stats.solution),
        tracer
    )
end
const newtoncg = solve!

"""
callback(nlp, solver, stats, logging)

Prints logging information for the Newton-CG solver.
    """
    function callback(
        nlp::KLLSModel{T},
        solver,
        M,
        trunk_stats,
        tracer,
        logging;
        atol::T = √eps(T),
        rtol::T = √eps(T),
        max_iter::Int = typemax(Int),
        trace::Bool = false
        ) where T
        
        dObj = trunk_stats.objective 
        iter = trunk_stats.iter
        r = trunk_stats.dual_feas # = ||∇ dual obj(x)|| = ||λy||
        # r = norm(solver.gx)
        Δ = solver.tr.radius
        actual_to_predicted = solver.tr.ratio
        cgits = solver.subsolver.stats.niter
        cgexit = cg_msg[solver.subsolver.stats.status]
        
        # Test exit conditions
        tired = iter >= max_iter
        optimal = r < atol + rtol * nlp.data.bNrm 
        done = tired || optimal
        
        log_items = (iter, dObj, r, Δ, actual_to_predicted, cgits, cgexit) 
        trace && push!(tracer, log_items)
        if logging > 0 && iter == 0
            # print a header with
            # - problem dimensions
            # Print a line of = signs
            println("  ","="^64)
            @printf("  KL-regularized least squares\n")
            @printf("  m = %5d  atol = %7.1e  bNrm = %7.1e\n", size(nlp.data.A, 1), atol, nlp.data.bNrm)
            @printf("  n = %5d  rtol = %7.1e     λ = %7.1e\n", size(nlp.data.A, 2), rtol, nlp.data.λ)
            println("  ","="^64,"\n")
            @printf("%8s   %9s   %9s   %9s   %9s   %6s   %10s\n",
            "iter","objective","∥λy∥","Δ","Δₐ/Δₚ","cg its","cg msg")
        end
        if logging > 0 && (mod(iter, logging) == 0 || done)
            @printf("%8d   %9.2e   %9.2e   %9.1e  %9.1e   %6d   %10s\n", (log_items...))
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