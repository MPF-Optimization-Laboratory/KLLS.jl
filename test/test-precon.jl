using KLLS, Test, LinearAlgebra, Random
import Krylov: cg

@testset failfast=true "Preconditioning" begin
    # Generate random data for tests
    Random.seed!(1234)
    m, n = 10, 30
    A = randn(m, n)
    H = A*A'
    d = randn(m)
    b = randn(10)
    λ = rand()
    data = KLLSModel(A, b, λ=λ)

    # To start: unconditioned CG on any PSD Hp=d system
    xi, sti = cg(H,d)
    Δ = 0.1*norm(xi)
    norm(H*xi-d) < 1e-10
    @test sti.status == "solution good enough given atol and rtol"

    # Construct an inverse
    Λ, U = eigen(H) # H = UΛU'
    M = U * pinv(diagm(Λ)) * U'; M = 0.5*(M+M')
    @test norm(M - H^-1) < 1e-10
    @test norm(M*H - I) < 1e-10
 
    # M ≈ inv(H), which requires ldiv=false
    Λ[1] = 0.5 
    M = U * pinv(diagm(Λ)) * U'; M = 0.5*(M+M')
    xm, stm = cg(H,d,M=M,ldiv=false)
    @test norm(H*xm-d) < 1e-10
    @test stm.status == "solution good enough given atol and rtol"
    @test all(xi .≈ xm)
    @test sti.niter > stm.niter

    # Diag(AA') preconditioner
    M = KLLS.DiagAAPreconditioner(data)
    P = Diagonal(diag(A*A'))
    @test all(P*d ≈ mul!(similar(d), M, d))
    @test all(P\d ≈ ldiv!(similar(d), M, d))
    xt, stt = cg(H,d,M=P,radius=Δ,verbose=0,ldiv=true)
    @test xt'*P*xt ≈ Δ^2

    # DiagASAtPreconditioner
    M = KLLS.DiagASAPreconditioner(data)
    g = KLLS.grad(data.kernel)
    S = Diagonal(g)
    P = Diagonal(A*S*A')
    @test all(P*d ≈ mul!(similar(d), M, d))
    @test all(P\d ≈ ldiv!(similar(d), M, d))
    xt, stt = cg(H,d,M=P,radius=Δ,verbose=0,ldiv=true)
    @test xt'*P*xt ≈ Δ^2

    # AA' preconditioner
    M = KLLS.AAPreconditioner(data)
    P = A*A' + λ*I
    @test all(P*d ≈ mul!(similar(d), M, d))
    @test all(P\d ≈ ldiv!(similar(d), M, d))
    xt, stt = cg(H,d,M=M,radius=Δ,verbose=0,ldiv=true)
    @test xt'*P*xt ≈ Δ^2
end