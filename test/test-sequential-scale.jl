using Test
using KLLS, NLPModels, LinearAlgebra, Random

@testset "SSModel SequentialSolve test case" begin
      Random.seed!(1234)
      tol = 2e-5
      λ = 1e-2
      m, n = 8, 10
      kl = KLLS.randKLmodel(m, n) 
      A, b = kl.A, kl.b
      regularize!(kl, λ)

      stats = solve!(kl)

      x = stats.solution
      r = stats.residual
      y = r/λ

      @test norm(A*x + r - b) < tol

      # test `residual!` (called by `residual`)
      ss = KLLS.SSModel(kl)
      Fy = residual!(ss, [y; 1], similar([y; 1]))
      @test norm(Fy[1:m]) < tol*kl.bNrm
      @test norm(Fy[1:m]) < stats.optimality*kl.bNrm
      Jy = jtprod_residual(ss, [y; 1], [Fy[1:m]; 0])
      @test norm(Jy) < tol

      # Form the Jacobian manually and compare
      t = 1.0
      scale!(kl, t)
      Jtrue = [ KLLS.dHess(kl) A*x; (A*x)' -1/t ]
      J = let
            e(i) = let e = zeros(m+1); e[i] = 1; e end
            reduce(hcat, [jtprod_residual(ss, [y; t], e(i)) for i in 1:m+1])
      end
      @test all( .≈(J, Jtrue, atol=1e-6) )

      rtol = tol
      atol = tol
      ssSoln = solve!(ss, SequentialSolve(), logging=0, atol=atol, rtol=rtol, zverbose=false)

      @test ssSoln.status == :optimal
end