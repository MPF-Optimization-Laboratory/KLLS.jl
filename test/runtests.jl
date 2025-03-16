using Test

@testset "test-perspectron-model" begin
    include("test-perspectron-model.jl")
end

@testset "test-level-set" begin
    include("test-level-set.jl")
end

@testset "test-linear-programming" begin
    include("test-linear-programming.jl")
end

@testset "test-newtoncg" begin
    include("test-newtoncg.jl")
end

@testset "test-optimal-transport" begin
    include("test-optimal-transport.jl")
end

@testset "test-precon" begin
    include("test-precon.jl")
end

@testset "test-selfscale" begin
    include("test-selfscale.jl")
end

@testset "test-sequential-scale" begin
    include("test-sequential-scale.jl")
end

# NonlinearSolve extension tests
# - removing because NonlinearSolve.jl is too heavy
# @testset "NonlinearSolve extension" begin
#     include("ext/nonlinearsolve_tests.jl")
# end