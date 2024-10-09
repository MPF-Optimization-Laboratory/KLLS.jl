"""
Dual objective:

    f(y) = λreg(A'y / λ) + 0.5 y∙Cy - b∙y

"""
function dObj!(cls::CLSModel, y)
    @unpack A, b, λ, C, reg, w = cls 
    increment!(cls, :neval_jtprod)
    w .= 0  # Reset buffer
    mul!(w, A', y ./ λ)  # w = A' * (y / λ)
    g = obj!(reg, w)
    return λ * g + 0.5 * dot(y, C * y) - dot(b, y)
end

NLPModels.obj(cls::CLSModel, y) = dObj!(cls, y)

"""
Dual objective gradient

   ∇f(y) = A∇reg(A'y / λ) + Cy - b 

evaluated at `y`. Assumes that the objective was last evaluated at the same point `y`.
"""
function dGrad!(cls::CLSModel, y, ∇f)
    @unpack A, b, λ, C, reg = cls
    increment!(cls, :neval_jprod)
    p = grad(reg)
    ∇f .= -b
    mul!(∇f, C, y, 1, 1)
    mul!(∇f, A, p, 1, 1)
    return ∇f
end

NLPModels.grad!(cls::CLSModel, y, ∇f) = dGrad!(cls, y, ∇f)

"""
Dual objective hessian

   ∇²f(y) = (1/λ)A∇²reg(A'y / λ)A' + C

evaluated at `y`. Assumes that the objective was last evaluated at the same point `y`.
"""
function dHess(cls::CLSModel)
    @unpack A, λ, C, reg = cls
    H = hess(reg)
    return (A*H*A') ./ λ + C
end

"""
    dHess_prod!(cls::CLSModel{T}, z, v) where T

Product of the dual objective Hessian with a vector `z`

    v ← ∇²d(y)z = (1/λ)A∇²reg(A'y / λ)A'z + Cz

where `y` is the point at which the objective was last evaluated.
"""
function dHess_prod!(cls::CLSModel, z, Hz)
    @unpack A, λ, C, w, reg = cls
    increment!(cls, :neval_jprod)
    increment!(cls, :neval_jtprod)
    H = hess(reg)
    mul!(w, A', z)                 # w = A'z
    mul!(w, H, w)                  # w = ∇²reg(A'y / λ)(A'z)
    mul!(Hz, A, w, 1 / λ, 0)       # v = (1/λ)A∇²reg(A'y / λ)A'z
    mul!(Hz, C, z, 1, 1)           # v += Cz
    return Hz
end

function NLPModels.hprod!(cls::CLSModel{T}, ::AbstractVector, z::AbstractVector, Hz::AbstractVector; obj_weight::Real=one(T)) where T
    return Hz = dHess_prod!(cls, z, Hz)
end
