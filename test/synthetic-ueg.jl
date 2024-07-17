import KLLS: DiagAAPreconditioner, KLLSModel, solve!, histogram
using LinearAlgebra, NPZ, StatsPlots
using UnPack

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
sP = solve!(klP, atol=1e-5, rtol = 1e-5, logging=1, trace=true)

# Get the solution
xP = sP.solution

# Report on the solution
histogram(sP, nbins=20, title="Histogram (log10 scale)", xscale=:log10, xlabel="")

# Create a new problem with a covariance matrix of `b`
C = inv.(b_std) |> diagm
klC = KLLSModel(A, b_avg, C=C, q=q, λ=1e-4)
# Solve the KL problem
sP = solve!(klC, atol=1e-5, rtol = 1e-5, logging=1, trace=true)
histogram(sP, nbins=20, title="Histogram (log10 scale)", xscale=:log10, xlabel="")
