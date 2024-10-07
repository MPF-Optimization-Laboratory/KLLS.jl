# Basic Interdependency Testing
# Making sure KLLS, optim, Matlab all run smoothly in one environment

using Optim
using DataFrames
using KLLS
using MATLAB
using Random
using NPZ
using CSV
using Dates
using LinearAlgebra

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

λ = 1e-3

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

   
function optimToDF(optimState::Vector)
    df = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[]) #, Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])
    for i in 1:size(optimState,1)
        log = (optimState[i].iteration, optimState[i].value, optimState[i].g_norm)
        push!(df,log)
    end

    return df
end

function dfToCSV(df::DataFrame,method_name::String)
    dt = Dates.now() 
    dt = Dates.format(dt, "mmdd_HHMM")
    filename = method_name * "_" * dt * ".csv"
    CSV.write("numerics/outputs/" * filename,df)

end


   
test = optimToDF(out.trace)
#dfToCSV(test,"test_method")


UEG_dict = npzread("data/synthetic-UEG_testproblem.npz")

q = convert(Vector{Float64}, UEG_dict["mu"])
q .= max.(q, 1e-13)
q .= q./sum(q)
C = inv.(UEG_dict["b_std"]) |> diagm
λ = 1e-4
n = length(q)

kl_UEG = KLLSModel(UEG_dict["A"],UEG_dict["b_avg"],C=C,q=q,λ=λ)






# no need to specify q, as default is uniform.


#=
#All of this faults. Will have to manually do matlab PDCO comparisons "offline"

mat"1+1"

x = range(-10.0, stop=10.0, length=500)
mat"version"

#mat"addpath('/Matlab/pdco/')"      
#testVar = mat"PDCO_KL.m"
=#