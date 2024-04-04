using KLLS
import Zygote
using LinearAlgebra, Random
using Convex, SCS

function logsumexp(z)
  m = maximum(z)
  return -m + log(sum(exp.(z .- m)))
end
function softmax(v,w)
  m = maximum(v)
  return exp.(v .- m) / sum(exp.(v .- m))
end

Random.seed!(123)
m, n = 4, 3
A = randn(m, n)
x = (v = rand(n); v ./ sum(v))
x̄ = ones(n)/n # (v = rand(n); v ./ sum(v))
b = A * x
λ = 1e-4
x0 = ones(n)/n
y0 = (b - A*x0)/λ

# Dual objective (min form)
f(y) = -dot(b, y) + λ*dot(y,y)/2 + logsumexp((A'y).*x̄)
g(y) = Zygote.gradient(f, y)[1]
H(y) = Zygote.hessian(f, y)

y = newton_opt(f, g, H, y0)

xfy(y) = (x̄.*exp.(A'y)) ./ sum( x̄.*exp.(A'y) )

xfy(y)