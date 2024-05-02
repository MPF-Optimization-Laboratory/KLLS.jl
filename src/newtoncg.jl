using NLPModels

struct KLLSData{T<:Real}
    A::AbstractMatrix{T}
    b::Vector{T}
    q::Vector{T}
    lse::LogExpFunction
    λ::T
end

function KLLSData(A, b; kwargs...)
    n = size(A, 2)
    q = fill(1/n, n)
    KLLSData(A, b, q; kwargs...)
end
function KLLSData(A, b, q; λ=1e-6)
    KLLSData(A, b, q, LogExpFunction(q), λ)
end

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
    @unpack A, b, λ, lse = nlp.data
    f, _, _ = lse(A'y)
    return f + 0.5λ * dot(y, y) - dot(b, y)
end

function NLPModels.grad!(nlp::KLLSModel, y::AbstractVector, ∇f::AbstractVector)
    @unpack A, b, λ, lse = nlp.data
    _, p, _ = lse(A'y)
    ∇f .= A*p + λ*y - b
end

function NLPModels.hprod!(nlp::KLLSModel, y::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(eltype(y)))
    @unpack A, λ, lse = nlp.data
    _, _, H = lse(A'y)
    Hz .= (A*(H*(A')*z)) + λ*z
end
