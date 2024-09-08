using Test, NPZ, UnPack, LinearAlgebra
using KLLS

data = try # needed because of vscode quirks while developing
    npzread("../data/synthetic-UEG_testproblem.npz")
catch
    npzread("./data/synthetic-UEG_testproblem.npz")
end

@unpack A, b_avg, b_std, mu = data
q = convert(Vector{Float64}, mu)
q .= max.(q, 1e-13)
q .= q./sum(q)

# Defaults:
# q = 1/n * ones(n) (uniform)
# λ = √ε
klP = KLLSModel(A, b_avg, q=q, λ=1e-4)

# Solve the KL problem
sP = solve!(klP, atol=1e-5, rtol = 1e-5, logging=0, trace=true)
@test sP.optimality < 1e-5*klP.bNrm

# Create a new problem with a covariance matrix of `b`
C = inv.(b_std) |> diagm
klC = KLLSModel(A, b_avg, C=C, q=q, λ=1e-4)

# Solve the KL problem
sP = solve!(klC, atol=1e-5, rtol = 1e-5, logging=0, trace=true)
@test sP.optimality < 1e-5*klC.bNrm

# Relax the simplex constraint to the nonnegative orthant.
t, xopt = maximize!(klC, zverbose=false, rtol=1e-6, logging=0, δ=1e-1)
@test KLLS.value!(klC, t) < 1e-6

scale!(klC, t)
sPt = solve!(klC, atol=1e-5, rtol = 1e-6, logging=0, trace=true)
r = sPt.residual
x = sPt.solution

@test norm(xopt - x) < 1e-5
@test norm(A*x + C*r - klC.b) < 1e-5

# Now use the self-scaling approach
scale!(klC, 1.0)
ss = KLLS.SSModel(klC)
xss, tss = solve!(ss, verbose=0, rtol=1e-6)

@test norm(xopt - xss) < 1e-5