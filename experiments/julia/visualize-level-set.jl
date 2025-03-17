# Level-Set Algorithm Visualization

# This script visualizes the level-set algorithm for solving the KL-regularized least squares problem:
#
#    (P) min_{p≥0} ϕ(p|q) + λ||y||² s.t. Ap + λy = b
#    (D) max_y b'y - λ||y||²_C - ϕ^*(A'y|q)
#
# where
# - p is the nonnegative n-dimensional measure 
# - q is the reference distribution
# - A is the constraint matrix
# - b is the target value
# - λ is the regularization parameter
# - y is the slack variable. Note that λy is the constraint residual; smaller λ enforces the constraint more tightly
#
# This visualization shows:
# 1. The value function v(t) at different scaling factors
# 2. The level-set algorithm's progress through animation
# 3. The geometric intuition of minorants used in the algorithm

# Ensure we're using the project's main environment first, then activate test env
import Pkg
# Get the absolute path to the project root directory
project_root = joinpath(@__DIR__, "..", "..")
# Activate the main project environment
Pkg.activate(project_root)
# Now activate the test environment which has Plots, DataFrames, etc.
import TestEnv; TestEnv.activate()

using DualPerspective, Plots, LinearAlgebra, UnPack, DataFrames
import NPZ: npzread
import JSOSolvers: TrunkSolver

# Load the synthetic UEG (Uniform Electron Gas) test problem data
# This data contains:
# - A: Constraint matrix
# - b_avg: Target values
# - b_std: Standard deviations for uncertainty
# - mu: Reference distribution
data_path = joinpath(project_root, "data", "synthetic-UEG_testproblem.npz")
data = npzread(data_path)
@unpack A, b_avg, b_std, mu = data
b = b_avg
q = convert(Vector{Float64}, mu)
q .= max.(q, 1e-13)  # Ensure positivity
q .= q./sum(q)      # Normalize to sum to 1
C = inv.(b_std) |> diagm  # Convert standard deviations to weights
λ = 1e-4  # Regularization parameter

# Create the DPModel using the data 
kl = DPModel(A, b, C=C, q=q, λ=λ)

## Sequential Solve - needed to get the target value σ
# This is the naive version of the level-set method based on
# rootfinding directly on the derivative of the value function
ssSoln = solve!(kl, SequentialSolve())
σ = -ssSoln.dual_obj  # Target value for visualization

# Visualize the value function v(t)
# Generate a range of scaling factors and compute v(t) for visualization
ts = range(0.5, 1.4, step=0.05)
vts = map(ts) do t
    scale!(kl, t)
    tSoln = solve!(kl)
    -tSoln.dual_obj
end 

# Reset the model for the level-set algorithm
kl = DPModel(A, b, C=C, q=q, λ=λ)

# Level-set algorithm parameters
mutable struct LevelSetState
    α::Float64    # Step size
    t::Float64    # Current scaling
    σ::Float64    # Target value
    tracer::DataFrame
end

state = LevelSetState(
    1.7,                    # α
    0.6,                    # Initial t
    σ,                      # Target value from earlier
    DataFrame(iter=Int[], l=Float64[], u=Float64[], u_over_l=Float64[], s=Float64[])
)

scale!(kl, state.t)
solver = TrunkSolver(kl)
frames = 5

# Animation state
path = [(state.t, σ)]  # Track (t,σ) pairs

anim = @animate for frame in 1:frames
    # Compute next iteration
    l, u, s = DualPerspective.oracle!(kl, state.α, state.σ, solver, state.tracer)
    t_next = state.t - l/s
    
    # Plot current state
    p = plot(ts, vts, label="v(t)", xlabel="t", ylabel="v(t)", 
            title="v(t) at iteration $frame", xlim=(0.5, 1.1), ylim=(-2, 40), 
            size=(900, 400))
    
    # Add current bounds and minorant
    scatter!(p, [state.t], [l], label="Lower bound", color=:green)
    scatter!(p, [state.t], [u], label="Upper bound", color=:blue)
    plot!(p, ts, s .* (ts .- state.t) .+ l, label="Lower minorant", linestyle=:dash)
    hline!(p, [state.σ], label="Target", color=:red)
    
    # Plot path of selected points
    push!(path, (t_next, state.σ))
    scatter!(p, first.(path), last.(path), label="Path", color=:red, 
            legend=:outerright)
    
    # Update state
    scale!(kl, t_next)
    state.t = t_next  # Update the mutable struct instead of creating a new one
end

# Save the animation
gif(anim, "anim_fps1.gif", fps = 1)

println("Visualization completed successfully.") 