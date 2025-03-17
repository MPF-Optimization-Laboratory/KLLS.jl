using DualPerspective
using Test
using NonlinearSolve
using LinearAlgebra
using Random

# Ensure reproducible tests
Random.seed!(42)

@testset "NonlinearSolve extension tests" begin
    # Create a simple test problem
    m, n = 10, 5
    A = randn(m, n)
    b = A * ones(n) + 0.1 * randn(m)  # Ensure feasible problem
    c = randn(n)
    λ = 1.0
    
    # Create the model
    model = DPModel(A, b, c=c, λ=λ)
    
    # Test NewtonEQ solver with default parameters
    stats = solve!(model, NewtonEQ())
    
    # Basic solution checks
    @test stats.status ∈ [:optimal, :max_iter]
    @test stats.elapsed_time > 0
    @test stats.iter > 0
    @test length(stats.solution) == n
    @test all(stats.solution .>= -1e-10)  # Allow for small numerical errors
    @test abs(sum(stats.solution) - 1.0) < 1e-6  # Should sum to 1
    
    # Test optimality conditions
    grad = A'*(A*stats.solution - b) + λ*c
    @test norm(grad) < 1e-4  # Gradient should be small at solution
    
    # Test with different parameters
    stats = solve!(model, NewtonEQ(), 
                  atol=1e-8, 
                  rtol=1e-8, 
                  max_iter=50,
                  trace=true)
    @test stats.status ∈ [:optimal, :max_iter]
    @test stats.iter ≤ 50
end 