# Preconditioning code
abstract type AbstractPreconditioner{T<:AbstractFloat} end
abstract type AbstractDiagPreconditioner{T} <: AbstractPreconditioner{T} end

# Extend the `mul!` and `ldiv!` functions to handle the `Preconditioner` type
import LinearAlgebra: mul!, ldiv!
mul!(z, P::AbstractDiagPreconditioner, v) = (z .= v .* P.d)
ldiv!(z, P::AbstractDiagPreconditioner, w) = (z .= w ./ P.d) 

# Default update routines
update!(P::AbstractDiagPreconditioner) = nothing
update!(P::UniformScaling) = nothing

######################################################
# Diagonal of Graham matrix:
#     M = Diag(diag(AA') + λ ))
#       = Diag(||aᵢ||² + λ, i∈[1,m])
######################################################
struct DiagAAPreconditioner{T, K<:KLLSModel{T}, V<:AbstractVector{T}} <: AbstractDiagPreconditioner{T}
    data::K
    d::V
end

"""
    DiagAAPreconditioner(data::KLLSData)

Construct a diagonal preconditioner for the KLLS problem with the matrix `A` and the vector `b`. The preconditioner is defined as

    M = Diag(diag(AA')) + λI,

where λ is the regularizer defined in the KLLSData object. The preconditioner is stored as a vector `d`. It's computed only once at construction.

See also [`mul!`](@ref), [`ldiv!`](@ref).
"""
function DiagAAPreconditioner(kl::KLLSModel)
    @unpack A, λ = kl
    d = map(a->dot(a,a)+λ, eachrow(A))
    DiagAAPreconditioner(kl, d)
end

######################################################
# Diagonal of weighted Graham matrix:
#     M = Diag(diag(ASA') + λ ))
# with
#     S = diagm(g) - gg'.
######################################################
struct DiagASAPreconditioner{T, K<:KLLSModel{T}, V<:AbstractVector{T}} <: AbstractDiagPreconditioner{T}
    kl::K
    d::V
end

"""
    DiagASAPreconditioner(data::KLLSData{T}; α::T=zero(T))

Construct a diagonal preconditioner for the KLLS problem with the matrix `A` and the vector `b`. The preconditioner is defined as

    M = Diag(diag(A(G-gg')A') + λ + α))

where G := diagm(g), and g is the LSE gradient at the current iterate. The preconditioner is stored as a vector `d`. It's computed at construction and at each call to `update!`

See also [`mul!`](@ref), [`ldiv!`](@ref), and [`update!`](@ref).
"""
function DiagASAPreconditioner(kl::KLLSModel)
    @unpack A, b, λ = kl
    d = diag_ASA!(similar(b), A, grad(kl.lse), λ)
    DiagASAPreconditioner(kl, d)
end

"""
    diag_ASA(d, A, g, λ) -> v

Compute the diagonals of the matrix

    M = Diag(diag(ASA') + λI ))

where S := diagm(g), and g is the LSE gradient. Thus,

    Diag(AGA')  = Diag(v), v = [<aᵢ,g∘a>, i∈[1,m]]
"""
function diag_ASA!(d, A, g, λ)
    for i in eachindex(d)
        for j in eachindex(g)
            d[i] = g[j]*A[i,j]^2
        end
    end
    if λ > 0
        @. d += λ
    end
    return d
end

function update!(P::DiagASAPreconditioner)
    d = P.d; A = P.kl.A; λ = P.kl.λ; g = grad(P.kl.lse)
    diag_ASA!(d, A, g, λ)
end

struct Preconditioner{Mat, Vec}
    M::Mat     # Preconditioner
    v::Vec     # buffer for linear solves
end

function Preconditioner(M)
    n = size(M, 1)
    v = Vector{eltype(M)}(undef, n)
    Preconditioner(M, v)
end

update!(P::Preconditioner) = nothing

function Base.show(io::IO, P::Preconditioner)
    println(io, "Preconditioner $(typeof(P.M))")
    println(io, "  size: $(size(P.M))")
end

"""
    mul!(z, M::Preconditioner{<:Cholesky}, d)

Calculate the product `z=M*d` with the Cholesky preconditioner `M = R'R` and the vector `d` and store in-place the result in `z`. The factor `R` is upper triangular. 
"""
function mul!(z, M::Preconditioner{<:Cholesky}, d)
   @unpack M, v = M
   R = M.U
   mul!(v, R, d)
   mul!(z, R', v) 
   return z
end

"""
   ldiv!(z, M::Preconditioner{<:Cholesky}, b)

Solve the linear system `z=M\b` with the Cholesky preconditioner `M = R'R` and the RHS `b` and store in-place the result in `z`. The factor `R` is upper triangular. 
"""
function ldiv!(z, P::Preconditioner{<:Cholesky}, b)
   @unpack M, v = P
   R = M.U
   ldiv!(v, R', b)
   ldiv!(z, R, v)
   return z
end

## Preconditioner for general matrix
mul!(z, P::Preconditioner{<:Matrix}, d) = mul!(z, P.M, d)
ldiv!(z, P::Preconditioner{<:Matrix}, b) = z .= P.M \ b

## Diagonal preconditioner
mul!(z, P::Preconditioner{<:Diagonal}, d) = mul!(z, P.M, d)
ldiv!(z, P::Preconditioner{<:Diagonal}, b) = ldiv!(z, P.M, b)