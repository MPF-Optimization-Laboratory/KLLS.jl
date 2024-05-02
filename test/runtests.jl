using Revise
using Random
using KLLS
using NLPModels
using JSOSolvers

# using Test
# @testset "KLLS.jl" begin
#     # Write your tests here.
# end

Random.seed!(1234)
m, n = 200, 300
# q = fill(1/n, n)
q = (v=rand(n); v/sum(v))
A = randn(m, n)
b = A*q + 0.1*randn(m)
λ = 1e-4

newton_opt(A, b, q, λ, max_iter=10000);

nlp = KLLS.KLLSModel(KLLSData(A, b, q, λ=λ))
stats = trunk(nlp, verbose=100)

# y = randn(m)
# obj(nlp, randn(m))
# grad!(nlp, y, zeros(m))
# hess_op!(nlp, y, randn(m), zeros(m))
