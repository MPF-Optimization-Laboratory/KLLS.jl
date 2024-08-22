using Test
using LinearAlgebra, Random
import Krylov: cg
using NLPModels
# using ADNLPModels, JSOSolvers
using KLLS

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
    g = KLLS.grad(data.lse)
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
    data = KLLSModel(A, b, λ=λ)
    atol = rtol = 1e-6
    st = solve!(data, atol=atol, rtol=rtol)
    x = st.solution; r = st.residual
    @test norm(A*x + r - b) < atol + rtol*norm(b)

    # Add preconditioning
    M = KLLS.AAPreconditioner(data)
    st = solve!(data, M=M, logging=0, atol=atol, rtol=rtol)

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

@testset "NLSModel" begin
    Random.seed!(1234)
    m, n = 2, 3
    kl = KLLS.randKLmodel(m, n)
    A = kl.A

    nl = KLLS.SSModel(kl)
    @test nl.meta.nvar == m+1

    ss = KLLS.SSModel(kl)
    @test ss.meta.nvar == m+1

    y, t = randn(m), 1.0
    yt = vcat(y, t)
    Fx = residual!(ss, yt, similar(yt))
   
    @test all(Fx[1:m] .== grad!(kl, y, similar(y) ))

    w = randn(m); α = randn(); x = KLLS.grad(kl.lse)
    Jyt = jprod_residual!(ss, yt, [w;α], similar([w;α]) )

    Hyw = KLLS.dHess_prod!(kl, w, copy(w))
    @test all( Hyw + (A*x)*α ≈ Jyt[1:m] )
    @test (A*x)'*w - α/t ≈ Jyt[end]

    # Compare J'r against ∇‖r(x)‖²
    Fx = residual(ss, yt)
    g1 = jtprod_residual!(ss, yt, Fx, similar([w;α]) )
    g2 = grad!(ss, yt, similar(yt))
    @test all(g1 .≈ g2)
end

@testset "Self-scaling" begin


end

@testset "Synthetic data" begin
    include("synthetic-ueg.jl")
end