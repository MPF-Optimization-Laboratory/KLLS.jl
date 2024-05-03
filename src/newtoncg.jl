using NLPModels
using JSOSolvers

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
    return dObj!(nlp.data, y)
end

function NLPModels.grad!(nlp::KLLSModel, y::AbstractVector, ∇f::AbstractVector)
    return dGrad!(nlp.data, y, ∇f)
end

function NLPModels.hprod!(nlp::KLLSModel, y::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(eltype(y)))
    return Hz = dHess_prod!(nlp.data, z, Hz)
end

function newtoncg(data::KLLSData; kwargs...)
    nlp = KLLSModel(data)
    stats = trunk(nlp; kwargs...) 
    p = grad(data.lse)
    y = stats.solution
    return p, y, stats
end