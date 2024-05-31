using KLLS
using LinearAlgebra, NPZ, StatsPlots

data = try # needed because of vscode quirks while developing
    npzread("../data/synthetic-UEG_testproblem.npz")
catch
    npzread("./data/synthetic-UEG_testproblem.npz")
end
kldata = KLLSData(data["A"], data["b_avg"])

# MAA = DiagAAPreconditioner(kldata; α=1e-3)
# MASA = DiagAAPreconditioner(kldata)

kldata.λ=1e-4

stats = solve!(kldata, atol=1e-5, rtol = 1e-5, logging=100, trace=true)
