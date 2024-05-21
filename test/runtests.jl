using Test
using LinearAlgebra, Random, Krylov
using KLLS

@testset "KLLSData correctness" begin
    Random.seed!(1234)
    m, n = 200, 400
    q = (v=rand(n); v/sum(v))
    A = randn(m, n)
    b = randn(m) 
    λ = 1e-3
    data = KLLSData(A, b, q=q, λ=λ)

    @test size(data.A) == (m, n)
    @test size(data.b) == (m,)
    @test size(data.q) == (n,)
    @test data.λ == λ
end

@testset "Preconditioning" begin

    Random.seed!(1234)

    # To start: unconditioned CG on any PSD Hp=d system
    m, n = 10, 30
    A = randn(m, n)
    H = A*A'
    d = randn(m)
    xi, sti = cg(H,d)
    norm(H*xi-d) < 1e-10
    @test sti.status == "solution good enough given atol and rtol"

    # Now preconditioned CG on the same system
    Λ, U = eigen(H) # H = UΛU'
    M = U * pinv(diagm(Λ)) * U'; M = 0.5*(M+M')
    @test norm(M - H^-1) < 1e-10
    @test norm(M*H - I) < 1e-10
 
    Λ[1] = 0.5 
    M = U * pinv(diagm(Λ)) * U'; M = 0.5*(M+M')
    xm, stm = cg(H,d,M=M)
    @test norm(H*xm-d) < 1e-10
    @test stm.status == "solution good enough given atol and rtol"
    @test all(xi .≈ xm)
    @test sti.niter > stm.niter

    # Create custom preconditioner
    P = KLLS.Preconditioner((M))
    @test all(M*d .≈ mul!(copy(d), P, d))
    @test all(M\d .≈ ldiv!(copy(d), P, d))
    P = KLLS.Preconditioner(cholesky(M))
    @test all(M*d .≈ mul!(copy(d), P, d))
    @test all(M\d .≈ ldiv!(copy(d), P, d))

    # # Add a radius
    Δ = 0.1*norm(xi)
    xt, stt = cg(H,d,M=KLLS.Preconditioner(M), radius=Δ, verbose=0)
    @test xt'*(M\xt) ≈ Δ^2
    xt, stt = cg(H,d,M=KLLS.Preconditioner(cholesky(M)), radius=Δ, verbose=0)
    @test xt'*(M\xt) ≈ Δ^2

end

@testset "Newton CG" begin
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    x0 = [1/i for i in 1:n]
    x0 = x0 / sum(x0)
    @test all(x0 .>= 0)
    @test sum(x0) ≈ 1.0

    b = randn(m)#A*x0
    λ = 1e-3
    data = KLLSData(A, b, λ=λ)
    atol = rtol = 1e-6
    x, y, st = newtoncg(data, atol=atol, rtol=rtol)
    @test norm(A*x + λ*y - b) < atol + rtol*norm(b)

    # Add preconditioning
    M = KLLS.Preconditioner(cholesky(A*A'))
    x, y, st = newtoncg(data, M=M, logging=1, atol=atol, rtol=rtol)

    cg(A*A', b)

end