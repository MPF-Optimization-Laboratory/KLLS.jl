# Basic Interdependency Testing
# Making sure KLLS, optim, Matlab all run smoothly in one environment

using Optim
using DataFrames
using KLLS
using MATLAB
using Random
using NPZ

#=
First set of tests, small A, random gaussian, uniform solution norm(x)=1
b = Ax
Solved via:
- KLLS (check, works as expected)
- BFGS (Works, some quirks with gradient in-place and paremeter ordering)

- MATLAB : The julia package is quite finicky, seems to constantly crash
=#

Random.seed!(1234)
m, n = 30, 50
A = randn(m, n)

x = [1/n for i in 1:n]
b = A*x

λ = 1e-3

data = KLLSModel(A, b, λ=λ)
atol = rtol = 1e-6

st = solve!(data, atol=atol, rtol=rtol,logging =1)
st.tracer


f(y) = KLLS.dObj!(data,y) # DUAL objective, dim y = dim b, duh

grad!(grad_in_place,y) = KLLS.dGrad!(data,y,grad_in_place)

function callbackFun(trace_value)
    return false
end

x0 = randn(m)
x0 = x0/sum(x0)

out =optimize(f,grad!,x0,
        LBFGS(),
        Optim.Options(
            store_trace = true,
            show_trace = false,
            callback = callbackFun
        )
        )

mat"1+1"

#=
x = range(-10.0, stop=10.0, length=500)
mat"version"

#mat"addpath('/Matlab/pdco/')"      
#testVar = mat"PDCO_KL.m"
=#