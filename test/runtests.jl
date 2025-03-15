using Test

@testset "test-klls-model" begin
    include("test-klls-model.jl")
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

# Extension tests
@testset "Extensions" begin
    # Test NonlinearSolve extension if available
    if Base.get_extension(KLLS, :NonlinearSolveExt) !== nothing
        @testset "NonlinearSolve extension" begin
            @info "Testing NonlinearSolve extension..."
            include("ext/nonlinearsolve_tests.jl")
        end
    else
        @info "Skipping NonlinearSolve extension tests (extension not available)"
    end
end