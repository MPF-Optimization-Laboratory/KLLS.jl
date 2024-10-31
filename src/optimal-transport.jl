struct OTModel{KL, T}
    m::Int64
    n::Int64
    ε::T
    λ::T
    kl::KL
end

function Base.show(io::IO, ot::OTModel)
    println(io, "Entropic Optimal Transport Model")
    println(io, @sprintf("   m = %10d  ϵ = %10.1e", ot.m, ot.ε))
    println(io, @sprintf("   n = %10d  λ = %10.1e", ot.n, ot.λ))
end

"""
    OTModel(C, p, q; kwargs...) -> OTModel

Construct a regularized optimal transport model with cost matrix `C` and marginals `p` and `q`.
"""
function OTModel_prior(p, q, C, ε; kwargs...)
    m, n = size(C)
    A = row_col_sum_operator(m, n)
    b = vcat(p, q)
    c = vec(C)
    q = inv.(exp.(c/ε))
    kl = KLLSModel(A, b, c=0*c, q=q; kwargs...)
    λ = kl.λ
    regularize!(kl, ε*kl.λ)
    return OTModel(m, n, ε, λ, kl)
end

function OTModel(p, q, C, ε; kwargs...)
    m, n = size(C)
    A = row_col_sum_operator(m, n)
    b = vcat(p, q)
    c = vec(C)/ε
    kl = KLLSModel(A, b, c=c; kwargs...)
    λ = kl.λ
    regularize!(kl, ε*kl.λ)
    return OTModel(m, n, ε, λ, kl)
end

function solve!(ot::OTModel; kwargs...)
    stats = solve!(ot.kl; kwargs...)
    P = reshape(stats.solution, ot.m, ot.n)
    return P, stats
end

function regularize!(ot::OTModel, λ)
   ot.kl.c ./= λ
   regularize!(ot.kl, λ)
end

"""
    row_col_sum_operator(m::Int, n::Int) -> LinearOperator

Create a linear operator that computes the row and column sums of the input
vector, reshaped as an m x n matrix.
"""
function row_col_sum_operator(m::Int, n::Int)
    N = m * n         # Size of the input vector x
    M = m + n         # Size of the output vector y
    en = ones(Float64, n)
    em = ones(Float64, m)

    """
    Define the forward map: y = α * (A * x) + β * y, where

        A(x) = [sum over rows; sum over columns], x reshaped as X = reshape(x, m, n)
    """
    function Afun!(y::AbstractVector{T}, x::AbstractVector{T}, α, β) where T
        @assert length(x) == N "Input vector x must have length m * n"
        @assert length(y) == M "Output vector y must have length m + n"

        if β == zero(T)
            y .= zero(T)
        else
            y .*= β
        end

        X = reshape(x, m, n)
        y1 = view(y, 1:m)
        y2 = view(y, m+1:m+n)

        # y1 is the row sums
        mul!(y1, X, en, α, one(T))          # y1 = α * X * en + y1

        # y2 is the col sums
        mul!(y2, adjoint(X), em, α, one(T)) # y2 = α * X' * em + y2
    end

    """
    Define adjoint operation: x = α * (A' * y) + β * x.
    """
    function Atfun!(x::AbstractVector{T}, y::AbstractVector{T}, α, β) where T
        # BLAS.ger! functions require the matrix and the scalar types to be the same
        # Separate the conversion logic from actual operation to minimize overhead
        α_T = convert(T, α)
        β_T = convert(T, β)

        _Atfun!(x, y, α_T, β_T)
    end

    function _Atfun!(x::AbstractVector{T}, y::AbstractVector{T}, α::T, β::T) where T
        @assert length(x) == N "Output vector x must have length m * n"
        @assert length(y) == M "Input vector y must have length m + n"

        if β == zero(T)
            x .= zero(T)
        else
            x .*= β
        end

        y1 = view(y, 1:m)
        y2 = view(y, m+1:m+n)

        X = reshape(x, m, n)

        BLAS.ger!(α, y1, en, X) # X = X + α(y1 * en')
        BLAS.ger!(α, em, y2, X) # X = X + α(em' * y2)
    end

    A = LinearOperator(Float64, M, N, false, false, Afun!, Atfun!)

    return A
end
