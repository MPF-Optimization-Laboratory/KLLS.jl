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

function newtoncg(data::KLLSData{T}; M=I, kwargs...) where T

    # Build the NLP model from the KL data
    nlp = KLLSModel(data)

    # Tracer
    tracer = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[], Δ=Float64[], Δₐ_Δₚ=Float64[], cgits=Int[], cgmsg=String[])

    # Callback routine
    cb(nlp, solver, stats) = callback(T, nlp, solver, stats, tracer; kwargs...)

    # Call the Trunk solver
    stats = trunk(nlp; M=M, callback=cb, atol=0., rtol=0.) 

    # Return the primal and dual solutions
    p = copy(grad(data.lse))
    y = copy(stats.solution)
    return p, y, stats, tracer
end

"""
    callback(nlp, solver, stats, logging)

Prints logging information for the Newton-CG solver.
"""
function callback(
    ttype::Type{T},
    nlp,
    solver,
    stats,
    tracer;
    atol::T = √eps(T),
    rtol::T = √eps(T),
    max_iter::Int = typemax(Int),
    logging::Int = 0,
    trace::Bool = false
    ) where T

    f = stats.objective 
    k = stats.iter
    r = stats.dual_feas # = ||∇ dual obj(x)|| = ||λy||
    # r = norm(solver.gx)
    Δ = solver.tr.radius
    actual_to_predicted = solver.tr.ratio
    cgits = solver.subsolver.stats.niter
    cgexit = cg_msg[solver.subsolver.stats.status]

    # Test exit conditions
    tired = k >= max_iter
    optimal = r < atol + rtol * nlp.data.bNrm  
    done = tired || optimal

    trace && push!(tracer, (k, f, r, Δ, actual_to_predicted, cgits, cgexit))
    if logging > 0 && k == 0
        # print a header with
        # - problem dimensions
        # Print a line of = signs
        println("  ","="^64)
        @printf("  KL-regularized least squares\n")
        @printf("  m = %5d\t atol = %9.1e\t  bNrm = %9.1e\n", size(nlp.data.A, 1), atol, nlp.data.bNrm)
        @printf("  n = %5d\t rtol = %9.1e\n", size(nlp.data.A, 2), rtol)
        println("  ","="^64,"\n")
        @printf("%8s   %9s   %9s   %9s   %9s   %6s   %10s\n",
                "iter","objective","∥λy∥","Δ","Δₐ/Δₚ","cg its","cg msg")
    end
    if logging > 0 && (mod(k, logging) == 0 || done)
        @printf("%8d   %9.2e   %9.2e   %9.1e  %9.1e   %6d   %10s\n", k, f, r, Δ, actual_to_predicted, cgits, cgexit)
    end

    if done
       stats.status = :user
       if logging > 0
           @printf("\n")
           if tired
               @printf("Maximum number of iterations reached\n")
           elseif optimal
               @printf("Optimality conditions satisfied\n")
           end
           @printf("Products with A   : %9d\n", neval_jprod(nlp))
           @printf("Products with A'  : %9d\n", neval_jtprod(nlp))
           @printf("Time elapsed (sec): %9.1f\n", stats.elapsed_time)
       end
    end
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