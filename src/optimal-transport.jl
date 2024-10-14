struct OTModel{KL}
    kl::KL
end

"""
    OTModel(C, p, q; kwargs...) -> OTModel

Construct a regularized optimal transport model with cost matrix `C` and marginals `p` and `q`.
"""
function OTModel(C, p, q; kwargs...)
    m, n = size(C)
    A = row_col_sum_operator(m, n)
    b = vcat(p, q)
    c = vec(C)
    kl = KLLSModel(A, b, c=c; kwargs...)
    return OTModel(kl)
end

function solve!(ot::OTModel; kwargs...)
    stats = solve!(ot.kl; kwargs...)
end
regularize!(ot::OTModel, λ) = regularize!(ot.kl, λ)

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