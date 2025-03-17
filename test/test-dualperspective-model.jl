using DualPerspective, Test, LinearAlgebra, Random

@testset "PTModel correctness" begin
    Random.seed!(1234)
    m, n = 200, 400
    q = (v=rand(n); v/sum(v))
    A = randn(m, n)
    b = randn(m) 
    λ = 1e-3
    data = PTModel(A, b, q=q, λ=λ)

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
    data = PTModel(A, b)

    # Test scaling
    @test try
        scale!(data, 0.5)
        true
    catch
        false
    end
    @test data.scale == 0.5

    # Test regularization
    @test try
        regularize!(data, 1e-3)
        true
    catch
        false
    end
    @test data.λ == 1e-3

    # Test initial guess update
    y0 = ones(Float64, m)
    @test try
        update_y0!(data, y0)
        true
    catch
        false
    end
    @test data.meta.x0 == y0
    
end
