using Test
using LinearAlgebra, Random
import Krylov: cg
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

    Random.seed!(1234)

    # To start: unconditioned CG on any PSD Hp=d system
    m, n = 10, 30
    A = randn(m, n)
    H = A*A'
    d = randn(m)
    xi, sti = cg(H,d)
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

    # Create custom preconditioner
    M = U * diagm(Λ) * U'; M = 0.5*(M+M')
    @testset for (f, Q) in zip([Matrix, cholesky, Diagonal], [M, M, Diagonal(H)])
        P = KLLS.Preconditioner(f(Q))
        @test all(Q*d ≈ mul!(similar(d), P, d))
        @test all(Q\d ≈ ldiv!(similar(d), P, d))
        xm, stm = cg(H,d,M=P,ldiv=true)
        @test stm.status == "solution good enough given atol and rtol"
        @test all(xi .≈ xm)
        @test sti.niter ≥ stm.niter
    end

    # # Add a radius
    Δ = 0.1*norm(xi)
    M = U * diagm(Λ) * U'; M = 0.5*(M+M')
    @testset for (f, Q) in zip([Matrix, cholesky, Diagonal], [M, M, Diagonal(H)])
        P = KLLS.Preconditioner(f(Q))
        xt, stt = cg(H,d,M=P,radius=Δ,verbose=0,ldiv=true)
        @test xt'*(Q*xt) ≈ Δ^2
    end

    # DiagAAtPreconditioner
    A = randn(10, 20)
    b = randn(10)
    λ = rand()
    data = KLLSModel(A, b, λ=λ)
    M = KLLS.DiagAAPreconditioner(data)
    P = Diagonal(diag(A*A')) + λ*I 
    @test all(P*d ≈ mul!(similar(d), M, d))
    @test all(P\d ≈ ldiv!(similar(d), M, d))

    # DiagASAtPreconditioner
    KLLS.dObj!(data, randn(size(A,1)))
    M = KLLS.DiagASAPreconditioner(data)
    g = KLLS.grad(data.lse)
    S = Diagonal(g) - g*g'
    P = Diagonal(A*S*A') + λ*I
    @test all(P*d ≈ mul!(similar(d), M, d))
    @test all(P\d ≈ ldiv!(similar(d), M, d))

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
    M = KLLS.Preconditioner(cholesky(A*A'))
    st = solve!(data, M=M, logging=0, atol=atol, rtol=rtol)

end

### TESTING TRUNK ###
# struct MyModel{T, S}<:AbstractNLPModel{T, S}
#     meta::NLPModelMeta{T, S}
#     counters::Counters
# end

# x0 = zeros(2)
# obj(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# # Objective and gradient
# NLPModels.obj(nlp::MyModel, x) = obj(x)
# function NLPModels.grad!(nlp::MyModel, x, ∇f)
#     ∇f .= [-2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2), 200*(x[2] - x[1]^2)]
# end

# # Hessian-vector product
# function NLPModels.hprod!(
#     nlp::MyModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(eltype(x)))
#     H = [2 - 400*(x[2]-x[1]^2) + 800*x[1]^2 -400*x[1]
#          -400*x[1]                            200]
#     Hv .= H*v
# end

# adnlp = ADNLPModel(obj, x0)
# nlp = MyModel(NLPModelMeta(2), Counters())
# stats = trunk(nlp, verbose=1)