# Preconditioning code
abstract type AbstractPreconditioner{T<:AbstractFloat} end
abstract type AbstractDiagPreconditioner{T} <: AbstractPreconditioner{T} end

# Extend the `mul!` and `ldiv!` functions to handle the `Preconditioner` type
import LinearAlgebra: mul!, ldiv!
mul!(z, P::AbstractDiagPreconditioner, v) = (z .= v .* P.d)
ldiv!(z, P::AbstractDiagPreconditioner, w) = (z .= w ./ P.d) 

# Default update routines
update!(::AbstractPreconditioner) = nothing
update!(::UniformScaling) = nothing

######################################################
# Diagonal of Graham matrix:
#     M = Diag(diag(AA') + λ ))
#       = Diag(||aᵢ||² + λ, i∈[1,m])
######################################################
struct DiagAAPreconditioner{T, K<:PTModel{T}, V<:AbstractVector{T}} <: AbstractDiagPreconditioner{T}
    kl::K
    d::V
end

"""
    DiagAAPreconditioner(data::PTModel)

Construct a diagonal preconditioner for the Perspectron problem with the matrix `A` and the vector `b`. The preconditioner is defined as

    M = Diag(diag(AA')) + λI,

where λ is the regularizer defined in the PTModel object. The preconditioner is stored as a vector `d`. It's computed only once at construction.
"""
# TODO: Verify that this is correct. At the very least, fix docstring above, which doesn't match the code.
function DiagAAPreconditioner(kl::PTModel{T}; α::T=zero(T)) where T
    d = map(a->dot(a,a)+α, eachrow(kl.A))
    DiagAAPreconditioner(kl, d)
end

######################################################
# Diagonal of weighted Graham matrix:
#     M = Diag(diag(ASA') + λ ))
# with
#     S = diagm(g) - gg'.
######################################################
struct DiagASAPreconditioner{T, K<:PTModel{T}, V<:AbstractVector{T}} <: AbstractDiagPreconditioner{T}
    kl::K
    d::V
    α::T
end

"""
    DiagASAPreconditioner(data::PTModel{T})

Construct a diagonal preconditioner for the KLLS problem with the matrix `A` and the vector `b`. The preconditioner is defined as

    M = Diag(diag(A(G-gg')A') + λ))

where G := diagm(g), and g is the LSE gradient at the current iterate. The preconditioner is stored as a vector `d`. It's computed at construction and at each call to `update!`

See also [`mul!`](@ref), [`ldiv!`](@ref), and [`update!`](@ref).
"""
function DiagASAPreconditioner(kl::PTModel{T}; α::T=zero(T)) where T
    @unpack A, b, λ = kl
    obj!(kl.lse, A'b)
    d = diag_ASA!(similar(b), A, grad(kl.lse), α)
    DiagASAPreconditioner(kl, d, α)
end

"""
    diag_ASA(d, A, g, λ) -> v

Compute the diagonals of the matrix

    M = Diag(diag(ASA') + λI ))

where S := diagm(g), and g is the LSE gradient. Thus,

    Diag(AGA')  = Diag(v), v = [<aᵢ,g∘a>, i∈[1,m]]
"""
function diag_ASA!(d::AbstractVector{T}, A, g, λ) where T
    for i in eachindex(d)
        d[i] = zero(T)
        for j in eachindex(g)
            d[i] += g[j]*A[i,j]^2
        end
    end
    if λ > 0
        @. d += λ
    end
    return d
end

function update!(P::DiagASAPreconditioner)
    d = P.d; A = P.kl.A; α = P.α; g = grad(P.kl.lse)
    diag_ASA!(d, A, g, α)
end

###############################
# Cholesky factorization of AA'
###############################

struct AAPreconditioner{T, K<:PTModel{T}, F<:Cholesky{T}, V<:AbstractVector} <: AbstractPreconditioner{T}
    kl::K
    P::F
    v::V # buffer for linear solves
end

function AAPreconditioner(kl::PTModel{T}) where T
    @unpack A, λ = kl
    v = Vector{T}(undef, size(A, 1))
    AAPreconditioner(kl, cholesky(A*A'+λ*I), v)
end

"""
    mul!(z, M::AAPreconditioner, d)

Calculate the product `z=M*d` with the Cholesky preconditioner `M = R'R` and the vector `d` and store in-place the result in `z`. The factor `R` is upper triangular. 
"""
function mul!(z, M::AAPreconditioner, d)
   @unpack P, v = M
   R = P.U
   mul!(v, R, d)
   mul!(z, R', v) 
   return z
end

"""
   ldiv!(z, M::AAPreconditioner, b)

Solve the linear system `z=M\b` with the Cholesky preconditioner `M = R'R` and the RHS `b` and store in-place the result in `z`. The factor `R` is upper triangular. 
"""
function ldiv!(z, M::AAPreconditioner, b)
   @unpack P, v = M
   R = P.U
   ldiv!(v, R', b)
   ldiv!(z, R, v)
   return z
end
