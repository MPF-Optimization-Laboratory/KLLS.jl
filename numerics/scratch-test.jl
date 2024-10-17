# Basic Interdependency Testing
# Making sure KLLS, optim, Matlab all run smoothly in one environment


using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates
using LinearAlgebra
using Plots

include("Optim.jl")

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

xs=[]
cb = tr -> begin
            push!(xs, [tr[end].metadata["f evals"] , tr[end].metadata["∇f evals"]])
            false
        end

out =Optim.optimize(f,grad!,z0,
        Optim.LBFGS(),
        Optim.Options(
            callback = cb,
            store_trace = true,
            show_trace = false,
            extended_trace = true
        )
        )

function optimToDF(optimState::Vector,cumulative_cost::Vector)
    df = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[],f_evals =Int[],grad_evals = []) #, Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])
    for i in 1:size(optimState,1)
        log = (optimState[i].iteration, optimState[i].value, optimState[i].g_norm,cumulative_cost[i][1],cumulative_cost[i][2])
        push!(df,log)
    end

    return df
end

d = optimToDF(out.trace,xs)

test_dict = npzread("data/MNIST_data_denoising.npz")

#q = convert(Vector{Float64}, UEG_dict["mu"])
#q .= max.(q, 1e-13)
#q.= q./sum(q)
#C = inv.(UEG_dict["b_std"]) |> diagm
λ = 1e-4
#n = length(q)

size(test_dict["A"]')
size(test_dict["b"])

kl_MNIST = KLLSModel(test_dict["A"]',test_dict["b"])
optTol = 10e-7

test_var = KLLS.solve!(kl_MNIST, atol=optTol, rtol=optTol,max_iter =200,trace=true,logging=true)

#=
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
=#

# no need to specify q, as default is uniform.


#=
#All of this faults. Will have to manually do matlab PDCO comparisons "offline"

mat"1+1"

x = range(-10.0, stop=10.0, length=500)
mat"version"

#mat"addpath('/Matlab/pdco/')"      
#testVar = mat"PDCO_KL.m"
=#

