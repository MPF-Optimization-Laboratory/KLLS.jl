# The Julia experiments have their own package environment
# Make sure to activate the environment before running these scripts
# import Pkg
# Pkg.activate("../../")
using KLLS, Plots, LinearAlgebra, NPZ, UnPack, DataFrames
import JSOSolvers

# Load the data
data = npzread("data/synthetic-UEG_testproblem.npz")
@unpack A, b_avg, b_std, mu = data
b = b_avg
q = convert(Vector{Float64}, mu)
q .= max.(q, 1e-13)
q .= q./sum(q)
C = inv.(b_std) |> diagm
λ = 1e-4
n = length(q)

kl = KLLSModel(A, b, C=C, c=zeros(n), q=q, λ=λ)

ssSoln = solve!(kl, SequentialSolve())

## Obtain the scaling factor
t = scale(kl)

σ=-ssSoln.dual_obj
ts = 0.5:0.05:1.4
vts = Float64[] 
for t in ts
    scale!(kl, t)
    tSoln = KLLS.solve!(kl)
    push!(vts, -tSoln.dual_obj)
end

kl = KLLSModel(A, b, C=C, c=zeros(n), q=q, λ=λ)

α=1.7
it = 0
tracer = DataFrame(iter=Int[], l=Float64[], u=Float64[], u_over_l=Float64[], s=Float64[])
l, u, s = 0.0, 0.0, 0.0
solver = JSOSolvers.TrunkSolver(kl)
start_time = time()
t0=0.6
scale!(kl, t0)
t=t0
frames = 5
accepted_ts = [t0]
ls = []
us = []
σs = [σ]

anim = @animate for i in 1:frames
    global it, l, u, s, t, tracer, accepted_ts, σs, kl, α, σ, solver, ts, vts 

    it += 1
    l, u, s = KLLS.oracle!(kl, α, σ, solver, tracer)  # TODO: weird max time
    plot(ts, vts, label="v(t)", xlabel="t", ylabel="v(t)", title="v(t) at iteration $it", xlim=(0.5, 1.1), ylim=(-2, 40), size=(900, 400))

    # Scatter points for bounds
    scatter!([t], [l], label="Lower bound", color=:green, marker=:circle, markersize=3)
    scatter!([t], [u], label="Upper bound", color=:blue, marker=:circle, markersize=3)

    # Calculate tk and plot minorant line
    tk = t - l / s
    line_y = [s * (t - kl.scale) + l for t in ts]

    # Horizontal line at y = σ
    hline!([σ], label="", color=:red, linestyle=:solid)

    # Plot line with slope s
    plot!(ts, line_y, label="Lower minorant", linestyle=:dash)

    # Add scatter points for selected scales
    push!(accepted_ts, tk)
    push!(σs, σ)

    # Update t and rescale
    t = tk
    scale!(kl, t)

    scatter!(accepted_ts[1:end], σs[1:end], label="Selected Scales", color=:red, marker=:circle, markersize=2, legend=:outerright)
    scatter!([accepted_ts[end]], [σs[end]], label="", color=:red, marker=:circle, markersize=3, legend=:outerright)
end
gif(anim, "anim_fps1.gif", fps = 1)