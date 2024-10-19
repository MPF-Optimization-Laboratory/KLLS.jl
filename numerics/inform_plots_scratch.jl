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

#laplace_dict = matread(joinpath(pwd(),"data","Laplace_inversion_large.mat"))
laplace_dict= matread(joinpath(pwd(),"data","rho-meson_testproblem.mat"))

A = laplace_dict["A"]
b = vec(laplace_dict["b_avg"])
x_test = vec(laplace_dict["x"])

#x_plot = vec(laplace_dict["x_values"])
#p_soln = vec(laplace_dict["p"])

μ = ones(size(A)[2]);
μ .= max.(μ, 1e-13)
μ .= μ./sum(μ)

optTol = 10e-12
max_iter =1000;
max_time = 60.0;
lambdas = (10.0).^(range(-12,stop=-9,length=4))
lambdas =round.(lambdas, sigdigits = 3)

cgitns = zeros(size(lambdas))


#################################################################################
#
# Figure 1 - ρ-meson
#
##########################################################################


figure_1 = plot(title = "ρ-meson problem", xlabel = "ω (frequency)", ylabel = "Probability")
plot!(x_test,labels = "True Density",lw = 2)

for (index,λ) in enumerate(lambdas)
    local data = KLLSModel(A,b,C=I, q = μ, λ = λ)
    global stats = KLLS.solve!(data, atol=optTol, rtol=optTol,max_iter = max_iter,trace=true,logging=true)
    plot!(stats.solution, labels = "λ = " * string(λ))
    cgitns[index] =sum(stats.tracer[:,6])
end


display(figure_1)

############################################################################
#
# figure 2 - UEG 
#
############################################################################

#=
UEG_dict= matread(joinpath(pwd(),"data","synthetic-UEG_testproblem.mat"))

A = UEG_dict["A"]
b = vec(UEG_dict["b_avg"])
x_test = vec(UEG_dict["x"])

μ = vec(UEG_dict["mu"])
μ .= max.(μ, 1e-13)
μ .= μ./sum(μ)
μ = convert(Vector{Float64},μ)

C = vec(inv.(UEG_dict["b_std"]')) |> diagm

optTol = 10e-12
max_iter =1000;
max_time = 60.0;
lambdas = (10.0).^(range(-4,stop=-2,length=3))
lambdas =round.(lambdas, sigdigits = 3)

############################################################################


figure_2 = plot(title = "UEG", xlabel = "", ylabel = "")
plot!(x_test,labels = "True Density")

for λ in lambdas
    local data = KLLSModel(A,b,C=C,q=μ,λ=λ)
    stats = KLLS.solve!(data, atol=optTol, rtol=optTol,max_iter = max_iter,trace=true,logging=true)
    plot!(stats.solution, labels = "λ = " * string(λ))

end


display(figure_2)
=#

############################################################################
#
# figure 2
#
#############################################################################

# One fixed λ to compare work between trunk / LBFGS / PDCO.
# pick a small one, 10e-4
#=
data = KLLSModel(A,b,C=I, q = μ, λ = 10e-4)

f(y) = KLLS.dObj!(data,y) # DUAL objective, 
grad!(grad_in_place,y) = KLLS.dGrad!(data,y,grad_in_place)
z0 = zeros(size(data.A,1))

opt_sol =Optim.optimize(f,grad!,z0,
    Optim.LBFGS(),
    Optim.Options(
        store_trace = true,
        show_trace = false,
        extended_trace = true,
        #callback = cb,
        # Piggyback trace to return ∇f and f evals per iteration
        g_tol = optTol + optTol*data.bNrm,
        # This is something to chat about, early stopping criterion
        f_tol = 0,
        x_tol = 0
)
)

Opt_sol.trace
=#