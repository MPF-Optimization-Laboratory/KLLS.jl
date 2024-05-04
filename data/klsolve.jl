using Revise
using KLLS
using UnPack
using NPZ
using Plots

# Read the data and create the KLLSData structure
data = npzread("PhysicsData.npz")
@unpack A, b, bn, x0 = data
klprob = KLLSData(A, b)

# Solve the problem over a range of regularization parameters λ

function solve!(klprob; λ=nothing, kwargs...)
    if λ != nothing
        klprob.λ = λ
    end
    p, y, stats = newtoncg(klprob; kwargs...)
    return p, stats.iter, stats.dual_feas
end 


p, y, stats = solve!(klprob, verbose=1)

exp10.(range(-6, stop=-3, length=k))


plot(title= "true and recovered distributions")
plot!(x0, label="true", lw=2) 
plot!(p, label="revcovered", lw=2)
