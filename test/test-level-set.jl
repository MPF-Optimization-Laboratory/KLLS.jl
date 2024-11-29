using KLLS, Test, LinearAlgebra, Random
using NPZ, UnPack

@testset "Level Set Method for random KLLSModel" begin
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    x0 = [1/i for i in 1:n]
    x0 = x0 / sum(x0)
    @test all(x0 .>= 0)
    @test sum(x0) ≈ 1.0

    b = randn(m)#A*x0
    λ = 1e-3
    data = KLLSModel(A, b, λ=λ)
    atol = rtol = 1e-6
    st = solve!(data, LevelSet(), atol=atol, rtol=rtol)
    x = st.solution; r = st.residual
    @test norm(A*x + r - b) < atol + rtol*norm(b)

    # Add preconditioning
    M = KLLS.AAPreconditioner(data)
    st = solve!(data, M=M, logging=0, atol=atol, rtol=rtol)
end

@testset "Level Set Method for KLLSModel with synthetic data" begin
    data = try # needed because of vscode quirks while developing
        npzread("../data/synthetic-UEG_testproblem.npz")
    catch
        npzread("./data/synthetic-UEG_testproblem.npz")
    end

    @unpack A, b_avg, b_std, mu = data
    b = b_avg
    q = convert(Vector{Float64}, mu)
    q .= max.(q, 1e-13)
    q .= q./sum(q)
    C = inv.(b_std) |> diagm
    λ = 1e-4
    n = length(q)

    # Create and solve the KL problem
    kl = KLLSModel(A, b, C=C, c=zeros(n), q=q, λ=λ)
    sP = solve!(kl, LevelSet(), atol=1e-5, rtol = 1e-5, logging=0, trace=true)
    @test sP.optimality < 1e-5*kl.bNrm
end
