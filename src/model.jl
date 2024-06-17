import NLPModels: NLPModels, NLPModelMeta, AbstractNLPModel, Counters, increment!, neval_jprod, neval_jtprod

"""
Structure for KLLS model 
    - `A` is the matrix of constraints
    - `b` is the right-hand side
    - `q` is the prior (default: uniform)
    - `lse` is the log-sum-exp function
    - `λ` is the regularization parameter
    - `w` is an n-buffer for the Hessian-vector product 
    - `bNrm` is the norm of the right-hand side
    - `name` is the name of the problem
"""
@kwdef mutable struct KLLSModel{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}, S} <: AbstractNLPModel{T, S}
    A::M = Matrix{Float64}(undef, 0, 0)
    b::V = Vector{Float64}(undef, 0)
    q::V = fill(eltype(A)(1/size(A, 2)), size(A, 2))
    λ::T = √eps(eltype(A))
    w::V = similar(q)
    bNrm::T = norm(b)
    scale::T = one(eltype(A))
    lse::LogExpFunction{V} = LogExpFunction(q)
    name::String = ""
    meta::NLPModelMeta{T, S} = NLPModelMeta(size(A,1), name="KLLS Model")
    counters::Counters = Counters()
end

KLLSModel(A, b; kwargs...) = KLLSModel(A=A, b=b; kwargs...)

function Base.show(io::IO, kl::KLLSModel)
    println(io, "KL regularized least-squares"*
                (kl.name == "" ? "" : ": "*kl.name))
    @printf("   m = %5d  bNrm = %7.1e\n", size(kl.A, 1), kl.bNrm)
    @printf("   n = %5d  λ    = %7.1e\n", size(kl.A, 2), kl.λ)
end

function scale!(kl::KLLSModel{T}, scale::T) where T
    kl.scale = scale
    return kl
end

function reset!(kl::KLLSModel)
    for f in fieldnames(Counters)
      setfield!(kl.counters, f, 0)
    end
    return kl
end