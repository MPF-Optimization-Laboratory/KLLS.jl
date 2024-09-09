using Test, NPZ, UnPack, LinearAlgebra
using KLLS

data = try # needed because of vscode quirks while developing
    npzread("../data/synthetic-UEG_testproblem.npz")
catch
    npzread("./data/synthetic-UEG_testproblem.npz")
end

@unpack A, b_avg, b_std, mu = data
b = b_avg
q = convert(Vector{Float64}, mu)
q .= max.(q, 1e-13)
q .= q./sum(q)
C = inv.(b_std) |> diagm
n = length(q)

# Create a new problem with a covariance matrix of `b`
kl = KLLSModel(A, b, C=C, c=zeros(n), q=q, λ=1e-4)

# Solve the KL problem
sP = solve!(kl, atol=1e-5, rtol = 1e-5, logging=0, trace=true)
@test sP.optimality < 1e-5*kl.bNrm

# Relax the simplex constraint to the nonnegative orthant.
t, xopt, jprods = maximize!(kl, zverbose=false, rtol=1e-6, logging=0, δ=1e-1)
@test KLLS.value!(kl, t) < 1e-6

scale!(kl, t)
sPt = solve!(kl, atol=1e-5, rtol = 1e-6, logging=0, trace=false)
r = sPt.residual
x = sPt.solution

@test norm(xopt - x) < 1e-5
@test norm(A*x + C*r - b) < 1e-5

# Now use the self-scaling approach
scale!(kl, 1.0)
ss = KLLS.SSModel(kl)
ssStats = solve!(ss, verbose=0, rtol=1e-6)
xss = ssStats.solution

@test norm(xopt - xss) < 1e-5