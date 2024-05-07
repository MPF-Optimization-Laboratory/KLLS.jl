using Revise
using KLLS
using NPZ
using Plots
using Printf
using UnPack
using LinearAlgebra
using Profile
data = npzread("./data/PhysicsData.npz", ["A", "b", "x0"])
@unpack A, b, x0 = data
klprob = KLLSData(A, b)
newtoncg(klprob)
klprob.λ = 0.0
@profview newtoncg(klprob, atol=1e-12, rtol=1e-12, verbose=0);




