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
b = vec(laplace_dict["b"])
x_plot = vec(laplace_dict["x_values"])
p_soln = vec(laplace_dict["p"])'

μ = ones(size(A)[2]);
μ .= max.(μ, 1e-13)
μ .= μ./sum(μ)

optTol = 10e-12
mas_iter =1000;
max_time = 60.0;
lambdas = (10.0).^(range(-6,stop=-1,length=6))
lambdas =round.(lambdas, sigdigits = 3)

#################################################################################
#
# Figure 1
#
##########################################################################


figure_1 = plot(title = "Laplace Inversion", xlabel = "", ylabel = "")
plot!(x_plot,p_soln',labels = "True Density")
for λ in lambdas
    local data = KLLSModel(A,b,C=I, q = μ, λ = λ)
    stats = KLLS.solve!(data, atol=optTol, rtol=optTol,max_iter = max_iter,trace=true)
    plot!(x_plot,stats.solution, labels = "λ = " * string(λ))
end

data = KLLSModel(A,b,C=I, q = μ, λ = λ)

f(y) = KLLS.dObj!(data,y) # DUAL objective, dim y = dim b, duh
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

primal_sol = KLLS.dGrad!(data,opt_sol.trace[end].metadata["x"]) + opt_sol.trace[end].metadata["g(x)"] + b

display(figure_1)