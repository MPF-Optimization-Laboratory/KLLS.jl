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
include("solve_metrics.jl")

lambdas = (10.0).^(range(-3,stop=3,length=6))

###########################################################
#
# UEG test problem
#
#############################################################
UEG_dict = npzread("data/synthetic-UEG_testproblem.npz")

q = convert(Vector{Float64}, UEG_dict["mu"])
q .= max.(q, 1e-13)
q .= q./sum(q)
C = inv.(UEG_dict["b_std"]) |> diagm
λ = 1e-4
n = length(q)

for λ in lambdas
    local kl_UEG = KLLSModel(UEG_dict["A"],UEG_dict["b_avg"],C=C,q=q,λ=λ)
    metrics(kl_UEG,"UEG")
end

##################################################################
#
# rho-meson test problem
#
##################################################################
rho_mes_dict = npzread("data/synthetic-UEG_testproblem.npz")
q = convert(Vector{Float64}, rho_mes_dict["mu"])
q .= max.(q, 1e-13)
q .= q./sum(q)
# This will be the uniform prior for rho-meson test problem, but included here for consistency

for λ in lambdas
    local kl_rho_mes = KLLSModel(A = rho_mes_dict["A"],b = rho_mes_dict["b_avg"],C=I,q=q, λ=λ)
    metrics(kl_rho_mes,"rho_meson")
end
