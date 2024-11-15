using Test

@testset "test-klls-model" begin
    include("test-klls-model.jl")
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