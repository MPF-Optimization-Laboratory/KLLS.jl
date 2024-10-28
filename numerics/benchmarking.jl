## Testing various problems 

###########################################################

using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates
using LinearAlgebra

include("solve_metrics.jl")

lambdas = (10.0).^(range(-5,stop=2,length=8))
lambdas = round.(lambdas,sigdigits = 3)
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
n = length(q)

for λ in lambdas
    local kl_UEG = KLLSModel(UEG_dict["A"],UEG_dict["b_avg"],C=C,q=q,λ=λ)
    solve_metrics(kl_UEG,"UEG")
end

##################################################################
#
# rho-meson test problem
#
##################################################################
rho_mes_dict = npzread("data/rho-meson_testproblem.npz")
q = convert(Vector{Float64}, rho_mes_dict["mu"])
q .= max.(q, 1e-13)
q .= q./sum(q)
# This will be the uniform prior for rho-meson test problem, but included here for consistency

for λ in lambdas
    local kl_rho_mes = KLLSModel(A = rho_mes_dict["A"],b = rho_mes_dict["b_avg"],C=I,q=q, λ=λ)
    try
        solve_metrics(kl_rho_mes,"rho_meson")
    catch
        print("fail case")
    end
end


##################################################################
#
# Imaging test problem (all 60k MNIST digits, handrawn 7)
#
##################################################################
MNIST_Dict = npzread("data/MNIST_data_denoising.npz")
q = ones(size( MNIST_Dict["A"])[1])
q .= q./sum(q)
#uniform prior 

for λ in lambdas
    local kl_MNIST = KLLSModel(A = MNIST_Dict["A"]',b = MNIST_Dict["b"],C=I,q=q, λ=λ)
    try
        solve_metrics(kl_MNIST,"MNIST denoising")
    catch
        print("Fail case")
    end
end