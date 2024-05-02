using Revise
using NLPModels, ADNLPModels
using JSOSolvers

struct MyModel{T, S}<:AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
end

x0 = zeros(2)
obj(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# Objective and gradient
NLPModels.obj(nlp::MyModel, x) = obj(x)
function NLPModels.grad!(nlp::MyModel, x, ∇f)
    ∇f .= [-2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2), 200*(x[2] - x[1]^2)]
end

# Hessian-vector product
function NLPModels.hprod!(
    nlp::MyModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(eltype(x)))
    H = [2 - 400*(x[2]-x[1]^2) + 800*x[1]^2 -400*x[1]
         -400*x[1]                            200]
    Hv .= H*v
end

adnlp = ADNLPModel(obj, x0)
nlp = MyModel(NLPModelMeta(2), Counters())
stats = trunk(nlp, verbose=1)
# trunk(adnlp, verbose=1)

# w = randn(2); v = randn(2)
# grad(adnlp, v) .≈ grad(nlp, v)
# hprod(adnlp, w, v)
# hprod(nlp, w, v)