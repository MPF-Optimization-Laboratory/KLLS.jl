## Testing various problems 

###########################################################

using Optim
using DataFrames
using KLLS
using MATLAB
using Random
using NPZ
using CSV
using Dates

##########################################################

include("metrics.jl")

###########################################################

UEG_dict = npzread("data/synthetic-UEG_testproblem.npz")

q = convert(Vector{Float64}, UEG_dict["mu"])
q .= max.(q, 1e-13)
q .= q./sum(q)
C = inv.(UEG_dict["b_std"]) |> diagm
λ = 1e-4
n = length(q)

kl_UEG = KLLSModel(UEG_dict["A"],UEG_dict["b_avg"],C=C,q=q,λ=λ)

metrics(kl_UEG,"UEG")


rho_mes_dict = npzread("data/synthetic-UEG_testproblem.npz")
kl_rho_mes = KLLSModel(A = rho_mes_dict["A"],b = rho_mes_dict["b_avg"])
# no need to specify q, as default is uniform.

metrics(kl_rho_mes,"rho_meson")
