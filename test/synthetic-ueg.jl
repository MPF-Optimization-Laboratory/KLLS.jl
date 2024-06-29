import KLLS: DiagAAPreconditioner, KLLSModel, solve!, histogram
using LinearAlgebra, NPZ, StatsPlots
using UnPack
using UnicodePlots

data = try # needed because of vscode quirks while developing
    npzread("../data/synthetic-UEG_testproblem.npz")
catch
    npzread("./data/synthetic-UEG_testproblem.npz")
end

@unpack A, b_avg, mu = data
q = convert(Vector{Float64}, mu)
q .= max.(q, 1e-13)
q .= q./sum(q)

# klU = KLLSModel(A, b_avg, λ=1e-6)
klP = KLLSModel(A, b_avg, q=q, λ=1e-4)

# MAA = DiagAAPreconditioner(klU)
# statsU = solve!(klU, atol=1e-5, rtol = 1e-5, logging=1, trace=true, M=MAA)

sP = solve!(klP, atol=1e-5, rtol = 1e-5, logging=1, trace=true)

x = sP.solution
print(sP)

histogram(sP, nbins=20, title="Histogram (log10 scale)", xscale=:log10, xlabel="")
