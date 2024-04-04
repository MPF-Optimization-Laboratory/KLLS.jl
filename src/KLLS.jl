module KLLS

using LinearAlgebra
using Printf
export newton_opt

struct LogExpFunction
    q::Vector{Float64}  # prior
    g::Vector{Float64}  # buffer for gradient
end

"""
Constructor for the LogExpFunction struct with prior `q`. 

    LogExpFunction(q) -> LogExpFunction(q, similar(q))
"""
function LogExpFunction(q::Vector) 
    @assert all(ζ->ζ≥0, q) && sum(q) ≈ 1
    LogExpFunction(q, similar(q))
end

"""
Constructor for the LogExpFunction struct with uniform prior.

    LogExpFunction(n) -> LogExpFunction(fill(1/n, n))
"""
LogExpFunction(n::Int) = LogExpFunction(fill(1/n, n))

"""
Evaluate logΣexp, its gradient, and Hessian at `p`:

    f, g, H = lse(p)

where `p` is a vector of length `n` and `lse` is a LogExpFunction object.

This implementation safeguards against numerical under/overflow,
and is based on the following code:

https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/4f9b688d298157dc24a5b0a518d971221fbe15dd/src/resampling.jl#L10
"""
function (lse::LogExpFunction)(p::Vector)
    w, q = lse.w, lse.q
    maxval, maxind = findmax(w)
    g .= exp.(q .* (p .- maxval))
    Σ = sum_all_but(g, maxind) # Σ = ∑gₑ-1
    f = log1p(Σ) + maxval
    g = g ./ (Σ+1)
    H = Diagonal
end

function sum_all_but(w, i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    s
end

"""
Newton's method for minimizing unconstrained optimization problems

    min_x KL(x|x̄) + λ/2 ||y||² subj to Ax + λy = b
    min_y logΣexp(A'y) - ⟨b, y⟩ +  λ/2 ||y||²

where `f` is a convex function, `A` is an m-by-n matrix, `b` is an m-vector, and `λ` is a nonnegative scalar.

# Input
- `A` is an m-by-n matrix
- `b` is an m-vector
- `x̄` is an n-vector on the simplex (no check is performed)
- `λ` is a nonnegative scalar (default: 0.0)
- `y0` is the initial guess (default: zeros(m))
- `optTol` is the relative tolerance for the gradient (default: 1e-6)
- `max_iter` is the maximum number of iterations (default: 100)
"""
function newton_opt(
    A::Matrix, b::Vector, x̄::Vector, λ::Real=0.0;
    y0::Vector = zeros(size(A, 2)),
    optTol::Real = 1e-6,
    max_iter::Int = 100)

    f, g, H = evalobj(fgh!, z)
    ϵ = optTol * (1 + norm(g)) # relative tolerance
    for k ∈ 1:max_iter
        @printf("%3d: %10.3e %10.3e %10.3e\n", k, f, norm(g), eigvals(H)[1])
        norm(g) < ϵ && break
        d = -(H \ g)
        if dot(d, g) > 0
            error("no descent")
        end
        @. z = z + d
        f, g, H = evalobj(fgh!, z)
    end
    return z
end

end # module