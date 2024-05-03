using Revise
using Random
using KLLS
using LinearAlgebra

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
λ = 1e-3
data = KLLSData(A, b, q, λ=λ)

pn, _, _ = newton_opt(data, max_iter=10000);
pc, yc, stats = newtoncg(data, verbose=100);

norm(pc)