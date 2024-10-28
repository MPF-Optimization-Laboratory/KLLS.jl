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

#using Optim
include("Optim.jl")
#include("utilities/trace.jl")

## Overwriting trace of optim package attempts
# Invalid redefinition of constant? What constant?
# Optim trace is a constant type not a mutable struct
#=
Optim.common_trace! = function(tr, d, state, iteration, method, options, curr_time=time())
    return trace(tr, d, state, iteration, method, options, curr_time=time())
end
=#

# Further ideas: Do OptimizationState have f_calls ? NO.
# 
#=
cb = tr -> begin
    push!(xs, tr[end].f_calls)
    false
end
=#

# Whatever "d" is in trace.jl has this info!




#Try with update?

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

    

out =Optim.optimize(f,grad!,z0,
        Optim.LBFGS(),
        Optim.Options(
            callback = cb,
            store_trace = true,
            show_trace = false,
            extended_trace = true
        )
        )
