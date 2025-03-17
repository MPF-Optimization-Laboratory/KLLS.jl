"""
Structure for Perspectron model 
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
@kwdef mutable struct PTModel{T<:AbstractFloat, M, CT, SB<:AbstractVector{T}, S<:AbstractVector{T}} <: AbstractNLPModel{T, S}
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
    mbuf::S = similar(b)
    mbuf2::S = similar(b)
    nbuf::S = similar(q)
    bNrm::T = norm(b)
    scale::T = one(eltype(A))
    lse::LogExpFunction = LogExpFunction(q)
    name::String = ""
    meta::NLPModelMeta{T, S} = begin
        m = size(A, 1)
        NLPModelMeta(m, name="Perspectron Model")
    end
    counters::Counters = Counters()
end

PTModel(A, b; kwargs...) = PTModel(A=A, b=b; kwargs...)

function Base.show(io::IO, kl::PTModel)
    println(io, "KL regularized least-squares"*
                (kl.name == "" ? "" : ": "*kl.name))
    println(io, @sprintf("   m = %10d  bNrm = %7.1e", size(kl.A, 1), kl.bNrm))
    println(io, @sprintf("   n = %10d  λ    = %7.1e", size(kl.A, 2), kl.λ))
    println(io, @sprintf("       %10s  τ    = %7.1e"," ", kl.scale))
end

"""
    regularize!(kl::PTModel{T}, λ::T) where T

Set the regularization parameter of the Perspectron model.
"""
function regularize!(kl::PTModel{T}, λ::T) where T
    kl.λ = λ
    return kl
end

"""
    scale(kl::PTModel)

Get the scaling factor of the Perspectron model.
"""
scale(kl::PTModel) = kl.scale

"""
    scale!(kl::PTModel{T}, scale::T) where T

Set the scaling factor of the Perspectron model.
"""
function scale!(kl::PTModel{T}, scale::T) where T
    kl.scale = scale
    return kl
end

function update_y0!(kl::PTModel{T}, y0::AbstractVector{T}) where T
    kl.meta = NLPModelMeta(kl.meta, x0=y0)
end

function NLPModels.reset!(kl::PTModel)
    for f in fieldnames(Counters)
      setfield!(kl.counters, f, 0)
    end
    return kl
end 