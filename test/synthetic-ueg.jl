import KLLS: DiagAAPreconditioner, KLLSData, solve!
using LinearAlgebra, NPZ, StatsPlots

data = try # needed because of vscode quirks while developing
    npzread("../data/synthetic-UEG_testproblem.npz")
catch
    npzread("./data/synthetic-UEG_testproblem.npz")
end
kl = KLLSModel(data["A"], data["b_avg"])

MAA = DiagAAPreconditioner(kl)
MASA = DiagAAPreconditioner(kldata)

kl.λ=1e-4

stats = solve!(kl, atol=1e-5, rtol = 1e-5, logging=100, trace=true)
