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

kl_UEG = KLLSModel(A = UEG_dict["A"],b = UEG_dict["b_avg"])
kl_UEG.q = UEG_dict["mu"]

metrics(kl_UEG,"UEG")


rho_mes_dict = npzread("data/synthetic-UEG_testproblem.npz")
kl_rho_mes = KLLSModel(A = rho_mes_dict["A"],b = rho_mes_dict["b_avg"])
# no need to specify q, as default is uniform.

metrics(kl_rho_mes,"rho_meson")
