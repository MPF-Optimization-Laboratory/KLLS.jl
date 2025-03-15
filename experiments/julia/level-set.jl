# Level-Set Algorithm Tests

# This script tests the level-set algorithm for solving the KL-regularized least squares problem and its dual:
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
# This script tests:
# 1. Sequential Solve method (naive rootfinding approach)
# 2. Adaptive Level-Set method (using minorants)

# Set up project environment by activating main project first,
# then switching to test environment for additional dependencies
import Pkg, TestEnv
project_root = joinpath(@__DIR__, "..", "..")
Pkg.activate(project_root)
TestEnv.activate()

using KLLS, LinearAlgebra, UnPack
import NPZ: npzread

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

# Create the KLLS model using the data 
kl = KLLSModel(A, b, C=C, q=q, λ=λ)

println("Testing level-set algorithms with UEG synthetic test problem...")

## Test 1: Sequential Solve

# This is the naive version of the level-set method based on
# rootfinding directly on the derivative of the value function
# to obtain v'(t) = 0.
println("\n## Testing Sequential Solve method")
ssSoln = solve!(kl, SequentialSolve())

# The scale is the sum of the primal variables.
# These values should be close to each other.
println("Scale: ", scale(kl))
println("sum(p): ", sum(ssSoln.solution))
println("Dual objective: ", ssSoln.dual_obj)

## Test 2: Level-Set Method using the minorant
println("\n## Testing Adaptive Level-Set method")
# Reset the model
kl = KLLSModel(A, b, C=C, q=q, λ=λ)
alsSoln = solve!(kl, AdaptiveLevelSet(), logging=1)

println("Scale: ", scale(kl))
println("sum(p): ", sum(alsSoln.solution))
println("Dual objective: ", alsSoln.dual_obj)

println("\nTests completed successfully.")


# Print comparison table
println("\nComparison of Methods:")
println("-"^72)
@printf("%-20s %12s %12s %12s %12s\n", "Method", "Mat-Vecs", "Feasibility", "Optimality", "Time (s)")
println("-"^72)
@printf("%-20s %12d %12.2e %12.2e %12.2f\n", 
    "Sequential", 
    ssSoln.neval_jprod + ssSoln.neval_jtprod,
    norm(ssSoln.residual, Inf),
    ssSoln.optimality,
    ssSoln.elapsed_time)
@printf("%-20s %12d %12.2e %12.2e %12.2f\n",
    "Adaptive Level-Set",
    alsSoln.neval_jprod + alsSoln.neval_jtprod,
    norm(alsSoln.residual, Inf),
    alsSoln.optimality,
    alsSoln.elapsed_time)
println("-"^72)