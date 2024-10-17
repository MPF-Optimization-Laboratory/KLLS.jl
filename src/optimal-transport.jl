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

    """
    Define the forward map: y = α * (A * x) + β * y, where

        Ax = A(X) = [Xe; X'e],  X = reshape(x, m, n)
    """
    function Afun!(y::AbstractVector{T}, x::AbstractVector{T}, α, β) where T
        @assert length(x) == N "Input vector x must have length m * n"
        @assert length(y) == M "Output vector y must have length m + n"

        if β == 0
            # If β == 0, we start with y = 0. Needed because y may be NaN
            fill!(y, zero(T))
        else
            # Scale the existing y by β
            y .*= β
        end

        # Compute y += α * (A * x)
        for j in 1:n
            for i in 1:m
                idx = (j - 1) * m + i  # Column-major order index
                val = α * x[idx]
                y[i] += val            # Accumulate row sum
                y[m + j] += val        # Accumulate column sum
            end
        end
    end

    """
    Define adjoint operation: y = α * (A' * y) + β * x.
    """
    function Atfun!(x::AbstractVector{T}, y::AbstractVector{T}, α, β) where T
        @assert length(x) == N "Output vector x must have length m * n"
        @assert length(y) == M "Input vector y must have length m + n"

        if β == zero(T)
            # If β == 0, we start with x = 0
            fill!(x, zero(T))
        else
            # Scale the existing x by β
            x .*= β
        end

        # Compute x += α * (A' * y)
        for j in 1:n
            y_col = y[m + j]
            for i in 1:m
                idx = (j - 1) * m + i  # Column-major order index
                x[idx] += α * (y[i] + y_col)
            end
        end
    end

    A = LinearOperator(Float64, M, N, false, false, Afun!, Atfun!)

    return A
end