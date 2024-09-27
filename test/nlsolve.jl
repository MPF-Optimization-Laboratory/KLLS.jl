using Test, NPZ, UnPack, LinearAlgebra
using NonlinearSolve, LinearSolve
using Suppressor
using KLLS

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
λ = 1e-3
m, n = size(A)

kl = KLLSModel(A, b, C=C, q=q, λ=λ)

# Now use the self-scaling approach
reset!(kl)
scale!(kl, 1.0)
ss = KLLS.SSModel(kl)
ssStats = KLLS.solve!(ss, verbose=0, rtol=1e-6)
xss = ssStats.solution

reset!(kl)
ff = NonlinearFunction(KLLS.nlresidual!; jvp=KLLS.nljprod!)
y0 = zeros(m)
yt0 = vcat(y0, 1.0)
prob = NonlinearProblem(ff, yt0, ss)
@suppress_err begin
sol = solve(
    prob,
    reltol=1e-6,
    abstol=1e-6,
    show_trace = Val(true),
    trace_level = TraceAll(),
    store_trace = Val(true),
    NewtonRaphson(
        linesearch = RobustNonMonotoneLineSearch(),
        linsolve = KrylovJL_MINRES(verbose=10, itmax=100, rtol=1e-4, atol=1e-4),
       ),
    )
end;

abstol = 1e-6
reltol = 1e-6
nlcache = init(
    prob,
    reltol=reltol,
    abstol=abstol,
    show_trace = Val(true),
    # trace_level = TraceAll(),
    # store_trace = Val(true),
    NewtonRaphson(
        linsolve = KrylovJL_MINRES(verbose=0, itmax=10, atol=abstol, rtol=reltol),
       ),
    )

for i in 1:10
    step!(nlcache)
end