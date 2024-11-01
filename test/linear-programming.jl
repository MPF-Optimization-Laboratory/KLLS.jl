using Test
using KLLS, LinearAlgebra, Random
using JuMP
using GLPK

@testset "Feasable LP" begin
    Random.seed!(1234)
    m = 50
    n = 100

    A = rand(m, n)

    x_feasible = rand(n)
    b = A * x_feasible

    c = rand(n)

    lp = KLLS.LPModel(A, b, c)
    stats = solve!(lp)
    optimal_x_lpmodel = stats.solution
    @test stats.status == :optimal

    model = Model(GLPK.Optimizer)

    @variable(model, x[1:n] >= 0)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .== b)

    optimize!(model)
    optimal_x_jump = value.(x)

    @test norm(optimal_x_jump .- optimal_x_lpmodel) < 1e-1
end

@testset "Infeasible LP" begin
    Random.seed!(1234)
    m = 50
    n = 100

    A = rand(m, n)

    x_feasible = rand(n)
    b = A * x_feasible

    # Modify b to make the problem infeasible
    # set the first row of A to zero and set a non-zero value in b
    A[1, :] .= 0.0
    b[1] = 1.0  # Since A[1, :] * x = 0 for any x, setting b[1] ≠ 0 makes it infeasible

    c = rand(n)

    lp = KLLS.LPModel(A, b, c)
    stats = solve!(lp)
    @test stats.status == :infeasible

    model = Model(GLPK.Optimizer)

    @variable(model, x[1:n] >= 0)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .== b)

    optimize!(model)

    status = termination_status(model)
    @test status == MOI.INFEASIBLE
end
