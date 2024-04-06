using KLLS

# using Test
# @testset "KLLS.jl" begin
#     # Write your tests here.
# end


m, n = 2, 3
q = fill(1/n, n)
A = randn(m, n)
b = A*q + 0.1*randn(m)
λ = 1e-2

newton_opt(A, b, q, λ);



