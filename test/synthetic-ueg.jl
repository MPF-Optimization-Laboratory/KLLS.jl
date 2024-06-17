import KLLS: DiagAAPreconditioner, KLLSModel, solve!
using LinearAlgebra, NPZ, StatsPlots
using UnPack

data = try # needed because of vscode quirks while developing
    npzread("../data/synthetic-UEG_testproblem.npz")
catch
    npzread("./data/synthetic-UEG_testproblem.npz")
end

@unpack A, b_avg, mu = data
q = convert(Vector{Float64}, mu)
q .= max.(q, 1e-13)
q .= q./sum(q)

klU = KLLSModel(A, b_avg, λ=1e-6)
klP = KLLSModel(A, b_avg, q=q, λ=1e-6)

MAA = DiagAAPreconditioner(klU)

statsU = solve!(klU, atol=1e-5, rtol = 1e-5, logging=1, trace=true, M=MAA)

# statsP = solve!(klP, atol=1e-5, rtol = 1e-5, logging=1, trace=true, M=MAA)

# plot([statsU.solution statsP.solution], label=["Uniform" "Prior"])
