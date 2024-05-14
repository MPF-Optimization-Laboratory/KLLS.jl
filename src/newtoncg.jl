import NLPModels: NLPModels, NLPModelMeta, AbstractNLPModel, Counters, increment!
import JSOSolvers: trunk

struct KLLSModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    data::KLLSData{T}
end

function Base.show(io::IO, data::KLLSData)
    println(io, "KLLS Data with $(size(data.A, 1)) rows and $(size(data.A, 2)) columns")
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

function newtoncg(data::KLLSData{T}; kwargs...) where T

    # Build the NLP model from the KL data
    nlp = KLLSModel(data)

    # Callback routine
    cb(nlp, solver, stats) = callback(T, nlp, solver, stats; kwargs...)

    # Call the Trunk solver
    # stats = tron(nlp; kwargs...) 
    stats = trunk(nlp; callback=cb, atol=0., rtol=0.) 

    # Return the primal and dual solutions
    p = copy(grad(data.lse))
    y = copy(stats.solution)
    return p, y, stats
end

"""
    callback(nlp, solver, stats, logging)

Prints logging information for the Newton-CG solver.
"""
function callback(
    ttype::Type{T},
    nlp,
    solver,
    stats;
    atol::T = √eps(T),
    rtol::T = √eps(T),
    max_iter::Int = typemax(Int),
    logging::Int = 0
    ) where T

    f = stats.objective 
    k = stats.iter
    r = stats.dual_feas # = ||∇ dual obj(x)|| = ||λy||
    # r = norm(solver.gx)
    Δ = solver.tr.radius
    cgits = solver.subsolver.stats.niter

    tired = k >= max_iter
    optimal = r < atol + rtol * max(1, norm(nlp.data.b))  
    done = tired || optimal

    if logging > 0 && k == 0
        # print a header with
        # - problem dimensions
        @printf("KLLS\n")
        @printf("m = %5d\t atol = %9.1e\n", size(nlp.data.A, 1), atol)
        @printf("n = %5d\t rtol = %9.1e\n", size(nlp.data.A, 2), rtol)
        @printf("\n")
        @printf("%8s   %9s   %9s   %9s   %6s\n",
                "iter","objective","∥λy∥","Δ","cg its")
    end
    if logging > 0 && (mod(k, logging) == 0 || done)
        @printf("%8d   %9.2e   %9.2e   %9.1e   %6d\n", k, f, r, Δ, cgits)
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
           @printf("Products with A   : %9d\n", nlp.counters.neval_jprod)
           @printf("Products with A'  : %9d\n", nlp.counters.neval_jtprod)
           @printf("Time elapsed (sec): %9.1f\n", stats.elapsed_time)
       end
    end
end