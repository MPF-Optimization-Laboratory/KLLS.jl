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
λ = 1e-4
n = length(q)

# Create and solve the KL problem
kl = KLLSModel(A, b, C=C, c=zeros(n), q=q, λ=λ)
sP = solve!(kl, atol=1e-5, rtol = 1e-5, logging=0, trace=true)
@test sP.optimality < 1e-5*kl.bNrm

# Value-function iteration: nonnegative 
reset!(kl)
t1, x1 = maximize!(kl, zverbose=true, rtol=1e-6, logging=0, δ=1e-1)
@test KLLS.value!(kl, t1) < 1e-6

# Solve the KL problem with the scaling `t1` obtained above
reset!(kl)
scale!(kl, t1)
sPt = solve!(kl, atol=1e-5, rtol = 1e-6, logging=0, trace=false)
x2, r2 = sPt.solution, sPt.residual
@test norm(x1 - x2) < 1e-5
@test norm(A*x2 + C*r2 - b) < 1e-5

# Now use the self-scaling approach
scale!(kl, 1.0)
ss = KLLS.SSModel(kl)
ssStats = solve!(ss, verbose=0, rtol=1e-6)
xss = ssStats.solution
@test norm(x2 - xss) < 1e-5
