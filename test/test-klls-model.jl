using KLLS, Test, LinearAlgebra, Random

@testset "KLLSData correctness" begin
    Random.seed!(1234)
    m, n = 200, 400
    q = (v=rand(n); v/sum(v))
    A = randn(m, n)
    b = randn(m) 
    λ = 1e-3
    data = KLLSModel(A, b, q=q, λ=λ)

    @test size(data.A) == (m, n)
    @test size(data.b) == (m,)
    @test size(data.q) == (n,)
    @test data.λ == λ
end

@testset "Modifiers" begin
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    b = randn(m)
    data = KLLSModel(A, b)
    @test try
        scale!(data, 0.5)
        true
    catch
        false
    end
    @test data.scale == 0.5
    @test try
        regularize!(data, 1e-3)
        true
    catch
        false
    end
    @test data.λ == 1e-3
end
