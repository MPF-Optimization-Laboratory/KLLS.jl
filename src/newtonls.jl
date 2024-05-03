"""
Newton's method for minimizing unconstrained optimization problems

    min_x KL(p|q) + λ/2 ||y||² subj to Ap + λq = b
    min_y logΣexp(A'y) - ⟨b, y⟩ +  λ/2 ||y||²

# Input
- `A` is an m-by-n matrix
- `b` is an m-vector
- `q` is an n-vector of priors in the unit simplex
- `λ` is a nonnegative scalar (default: 1e-6)
- `y0` is the initial guess (default: zeros(m))
- `optTol` is the relative tolerance for the gradient (default: 1e-6)
- `max_iter` is the maximum number of iterations (default: 100)

# Output
- `(p, y)` approximate primal-dual pair
"""
function newton_opt(
    data::KLLSData;
    y0::Vector = zeros(size(data.A, 1)),
    optTol::Real = 1e-6,
    max_iter::Int = 100,
    μ::Real = 1e-4)

    tracer = Tracer()
    y = copy(y0)
    ls_its = 0

    evaldual(y) = begin
        dObj = dObj!(data, y) 
        dGrd = dGrad!(data, y, similar(y))
        dHes = dHess(data)
        return dObj, dGrd, dHes
    end

    dObj, dGrd, dHes = evaldual(y)
    ϵ = optTol * (1 + norm(dGrd, Inf)) # relative tolerance

    for k ∈ 0:max_iter

        # Newton direction
        d = -((dHes+1e-2I) \ dGrd)

        # Log and test for exit
        logger!(tracer, k, dObj, dGrd, dHes, ls_its, d)
        if norm(dGrd, Inf) < ϵ
            break
        end

        # Newton step. Replace with a CG solver for large problems
        slope = dot(d, dGrd)
        if slope > 0
            error("no descent")
        end

        # Line search
        # suff_descent = dot(dGrd, d) ≤ -(1e-4)norm(d)^(2.1)
        # if !suff_descent
        #     d = -dGrd
        # end
        α, ls_its = armijo(y->evaldual(y)[1], dGrd, y, d, μ=μ)

        # Update y and evaluate objective quantities
        @. y = y + α*d
        dObj, dGrd, dHes = evaldual(y)

    end
    return grad(data.lse), y, tracer
end

function armijo(f, ∇fx, x, d; μ=1e-5, α=1, ρ=0.5, maxits=10)
    for k in 1:maxits
       if f(x+α*d) < f(x) + μ*α*dot(∇fx,d)
           return α, k
       end
       α *= ρ
    end
    error("$(dot(∇fx,d))   backtracking linesearch failed")
end

function logger!(tracer, k, dObj, dGrd, dHes, ls_its, d)
    nrmdGrd = norm(dGrd)
    F = eigen(Symmetric(dHes))
    λmin, λmax = F.values[1], F.values[end]
    if k == 0
       @printf("%4s: %11s %11s %8s %11s %11s %11s\n", "iter", "obj", "|grd|", "line its", "lmin", "lmax", "d⋅dG")
    end
    @printf("%4d: %11.4e %11.4e %8d %11.4e %11.4e %11.4e\n", k, dObj, nrmdGrd, ls_its, λmin, λmax, dot(dGrd, d))
    push!(tracer, dObj, nrmdGrd)
end