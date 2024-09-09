"""
Structure for KLLS model 
    - `A` is the matrix of constraints
    - `b` is the right-hand side
    - `q` is the prior (default: uniform)
    - `lse` is the log-sum-exp function
    - `λ` is the regularization parameter
    - `C` is a positive definite scaling matrix
    - `w` is an n-buffer for the Hessian-vector product 
    - `bNrm` is the norm of the right-hand side
    - `name` is the name of the problem
"""
@kwdef mutable struct KLLSModel{T<:AbstractFloat, M<:AbstractMatrix{T}, CT, SB<:AbstractVector{T}, S<:AbstractVector{T}} <: AbstractNLPModel{T, S}
    A::M
    b::SB
    c::S = begin
              m, n = size(A)
              c = ones(eltype(A), n)
            end
    q::S = begin
             m, n = size(A)
             q = similar(b, n)
             q .= 1/n
           end
    λ::T = √eps(eltype(A))
    C::CT = I
    w::S = similar(q)
    bNrm::T = norm(b)
    scale::T = one(eltype(A))
    lse::LogExpFunction = LogExpFunction(q)
    name::String = ""
    meta::NLPModelMeta{T, S} = begin
        m = size(A, 1)
        NLPModelMeta(m, name="KLLS Model")
    end
    counters::Counters = Counters()
end

KLLSModel(A, b; kwargs...) = KLLSModel(A=A, b=b; kwargs...)

function Base.show(io::IO, kl::KLLSModel)
    println(io, "KL regularized least-squares"*
                (kl.name == "" ? "" : ": "*kl.name))
    println(io, @sprintf("   m = %5d  bNrm = %7.1e", size(kl.A, 1), kl.bNrm))
    println(io, @sprintf("   n = %5d  λ    = %7.1e", size(kl.A, 2), kl.λ))
    println(io, @sprintf("       %5s  τ    = %7.1e"," ", kl.scale))
end

function regularize!(kl::KLLSModel{T}, λ::T) where T
    kl.λ = λ
    return kl
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