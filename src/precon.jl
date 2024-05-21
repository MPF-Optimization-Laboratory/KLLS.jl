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
    mul!(z, M::Preconditioner{<:Cholesky}, b)

Multiply the preconditioner `M = inv(R'R)` by the vector `b` and store the result in `z`. Here, `R` is triangular. 
"""
function mul!(z, M::Preconditioner{<:Cholesky}, b)
   @unpack M, v = M
   R = M.U
   ldiv!(v, R', b)
   ldiv!(z, R, v) 
   return z
end

"""
   ldiv!(z, M::Preconditioner{<:Cholesky}, b)

Compute `z=M\b` with the preconditioner `M = inv(R'R)` and the RHS `b` and store the result in `z`. Here, `R` is triangular. 
"""
function ldiv!(z, P::Preconditioner{<:Cholesky}, x)
   @unpack M, v = P
   R = M.U
   mul!(v, R, x)
   mul!(z, R', v)
   return z
end

## Preconditioner for general matrix
mul!(z, P::Preconditioner{<:Matrix}, x) = z .= P.M * x
ldiv!(z, P::Preconditioner{<:Matrix}, b) = z .= P.M \ b
