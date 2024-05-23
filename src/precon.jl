# Preconditioning code

struct Preconditioner{Mat, Vec}
    M::Mat     # Preconditioner
    v::Vec     # buffer for linear solves
end

function Preconditioner(M)
    n = size(M, 1)
    v = Vector{eltype(M)}(undef, n)
    Preconditioner(M, v)
end

function Base.show(io::IO, P::Preconditioner)
    println(io, "Preconditioner $(typeof(P.M))")
    println(io, "  size: $(size(P.M))")
end

# Extend the `mul!` and `ldiv!` functions to handle the `Preconditioner` type
import LinearAlgebra: mul!, ldiv!

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