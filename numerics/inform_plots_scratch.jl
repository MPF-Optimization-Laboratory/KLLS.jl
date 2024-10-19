## Informs Plots

using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates
using Plots
using MAT
using LinearAlgebra

include("plot_utils.jl")
include("solve_metrics.jl")

##################################################
#
# Data import
#
##################################################

laplace_dict = matread(joinpath(pwd(),"data","Laplace_inversion.mat"))

A = laplace_dict["A"]
b = laplace_dict["b"]
x_plot = laplace_dict["x_values"]
p_soln = laplace_dict["p"]'

μ = ones(size(A)[2]);
print(μ)
μ .= max.(μ, 1e-13)
μ .= μ./sum(μ)
μ = convert(Vector{Float64}, μ)

optTol = 10e-6
mas_iter =1000;
max_time = 60.0;
lambdas = (10.0).^(range(-6,stop=-1,length=6))
lambdas =round.(lambdas, sigdigits = 3)

#################################################################################
#
# Experiments
#
##########################################################################

solns =[]

for λ in lambdas
    local data = KLLSModel(A, b)
    #test = KLLS.solve!(data, atol=optTol, rtol=optTol,max_iter = max_iter,trace=true)
end

################################################################################
#
# Plotting.
#
#################################################################################