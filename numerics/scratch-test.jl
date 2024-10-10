# Basic Interdependency Testing
# Making sure KLLS, optim, Matlab all run smoothly in one environment

using Optim
using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates
using LinearAlgebra
using Plots
#=
First set of tests, small A, random gaussian, uniform solution norm(x)=1
b = Ax
Solved via:
- KLLS (check, works as expected)
- BFGS (Works, some quirks with gradient in-place and paremeter ordering)

- MATLAB : The julia package is quite finicky, seems to constantly crash
=#

Random.seed!(1234)
m, n = 60, 200
A = randn(m, n)

x0 = randn(n)
x0 = x0/sum(x0)
b = A*x0

λ = 1e-2

data = KLLSModel(A, b, λ=λ)
atol = rtol = 1e-6
st = KLLS.solve!(data, atol=atol, rtol=rtol,trace=true)


################################################################

f(y) = KLLS.dObj!(data,y) # DUAL objective, dim y = dim b, duh

grad!(grad_in_place,y) = KLLS.dGrad!(data,y,grad_in_place)


z0 = randn(m)

out =optimize(f,grad!,z0,
        LBFGS(),
        Optim.Options(
            store_trace = true,
            show_trace = false,
            extended_trace = true
        )
        )


UEG_dict = npzread("data/synthetic-UEG_testproblem.npz")

q = convert(Vector{Float64}, UEG_dict["mu"])
q .= max.(q, 1e-13)
q .= q./sum(q)
C = inv.(UEG_dict["b_std"]) |> diagm
λ = 1e-4
n = length(q)

kl_UEG = KLLSModel(UEG_dict["A"],UEG_dict["b_avg"],C=C,q=q,λ=λ)
optTol = 10e-7
test_var = KLLS.solve!(kl_UEG, atol=optTol, rtol=optTol,max_iter =200,trace=true)


FOLDER = "1010_1645"

files_to_plot = readdir(joinpath(@__DIR__ ,"outputs",FOLDER))
#print(files_to_plot)

for filename in files_to_plot
    UEG_plot = plot!(title = "UEG Residual", xlabel = "Iteration", ylabel = "r")
    if(occursin("UEG",filename))
        df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
        plot!(df[:,1],df[:,3], labels=[split(filename,"lam")[2]], legend=:topleft)
    end
    display(UEG_plot)
end


# no need to specify q, as default is uniform.


#=
#All of this faults. Will have to manually do matlab PDCO comparisons "offline"

mat"1+1"

x = range(-10.0, stop=10.0, length=500)
mat"version"

#mat"addpath('/Matlab/pdco/')"      
#testVar = mat"PDCO_KL.m"
=#