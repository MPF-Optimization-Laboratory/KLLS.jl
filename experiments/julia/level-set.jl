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

using KLLS, LinearAlgebra, UnPack, Printf
import NPZ: npzread
using UnicodePlots  # For Unicode plots

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

println("Testing level-set algorithms with UEG synthetic test problem...")

# Single λ comparison to show detailed information
λ_single = 1e-4  # Regularization parameter for detailed comparison
println("\n# Detailed comparison for λ = $(λ_single)")

# Create the KLLS model using the data 
kl = KLLSModel(A, b, C=C, q=q, λ=λ_single)

## Test 1: Sequential Solve with single λ
println("\n## Testing Sequential Solve method")
ssSoln = solve!(kl, SequentialSolve())

# The scale is the sum of the primal variables.
# These values should be close to each other.
println("Scale: ", scale(kl))
println("sum(p): ", sum(ssSoln.solution))
println("Dual objective: ", ssSoln.dual_obj)

## Test 2: Level-Set Method using the minorant with single λ
println("\n## Testing Adaptive Level-Set method")
# Reset the model
kl = KLLSModel(A, b, C=C, q=q, λ=λ_single)
alsSoln = solve!(kl, AdaptiveLevelSet(), logging=1)

println("Scale: ", scale(kl))
println("sum(p): ", sum(alsSoln.solution))
println("Dual objective: ", alsSoln.dual_obj)

println("\nTests completed successfully.")

# Print comparison table for single λ
println("\nComparison of Methods (λ = $(λ_single)):")
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

# Multi-λ comparison across a range of regularization parameters
println("\n\n# Comparison across multiple λ values")

# Generate logarithmically spaced λ values from 1e-1 to 1e-6
λ_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
n_λ = length(λ_values)

# Arrays to store results
sequential_feasibility = zeros(n_λ)
sequential_optimality = zeros(n_λ)
sequential_time = zeros(n_λ)
sequential_matvecs = zeros(Int, n_λ)

adaptive_feasibility = zeros(n_λ)
adaptive_optimality = zeros(n_λ)
adaptive_time = zeros(n_λ)
adaptive_matvecs = zeros(Int, n_λ)

# Run both methods for each λ value
for (i, λ) in enumerate(λ_values)
    println("\nTesting with λ = $λ")
    
    # Sequential Solve method
    kl_ss = KLLSModel(A, b, C=C, q=q, λ=λ)
    ss_result = solve!(kl_ss, SequentialSolve(), logging=0)
    
    sequential_feasibility[i] = norm(ss_result.residual, Inf)
    sequential_optimality[i] = ss_result.optimality
    sequential_time[i] = ss_result.elapsed_time
    sequential_matvecs[i] = ss_result.neval_jprod + ss_result.neval_jtprod
    
    # Adaptive Level-Set method
    kl_als = KLLSModel(A, b, C=C, q=q, λ=λ)
    als_result = solve!(kl_als, AdaptiveLevelSet(), logging=0)
    
    adaptive_feasibility[i] = norm(als_result.residual, Inf)
    adaptive_optimality[i] = als_result.optimality
    adaptive_time[i] = als_result.elapsed_time
    adaptive_matvecs[i] = als_result.neval_jprod + als_result.neval_jtprod
    
    # Print comparison table for this λ
    println("\nComparison of Methods (λ = $λ):")
    println("-"^72)
    @printf("%-20s %12s %12s %12s %12s\n", "Method", "Mat-Vecs", "Feasibility", "Optimality", "Time (s)")
    println("-"^72)
    @printf("%-20s %12d %12.2e %12.2e %12.2f\n", 
        "Sequential", 
        sequential_matvecs[i],
        sequential_feasibility[i],
        sequential_optimality[i],
        sequential_time[i])
    @printf("%-20s %12d %12.2e %12.2e %12.2f\n",
        "Adaptive Level-Set",
        adaptive_matvecs[i],
        adaptive_feasibility[i],
        adaptive_optimality[i],
        adaptive_time[i])
    println("-"^72)
end

# Create Unicode plots for comparing metrics vs λ

"""
    create_comparison_plot(λ_values, seq_data, adap_data, metric_name; width=80, height=20)

Create a log-log plot comparing Sequential and Adaptive methods across lambda values.

# Arguments
- `λ_values`: Array of lambda values
- `seq_data`: Array of data for Sequential method
- `adap_data`: Array of data for Adaptive method
- `metric_name`: String name of the metric being compared
- `width`: Plot width (default: 80)
- `height`: Plot height (default: 20)

# Returns
A UnicodePlots scatterplot object
"""
function create_comparison_plot(λ_values, seq_data, adap_data, metric_name; width=80, height=20)
    # Create the title and labels
    title = "$metric_name vs λ"
    
    # Create the initial plot with Sequential method data
    p = scatterplot(
        λ_values, 
        seq_data,
        title=title,
        xlabel="λ",
        ylabel="$metric_name (log)",
        xscale=:log10,
        yscale=:log10,
        name="Sequential",
        width=width, 
        height=height,
        marker='●',  # Larger circle character
        xflip=true   # Flip x-axis to show decreasing values from left to right
    )
    
    # Add Adaptive method data to the plot
    p = scatterplot!(
        p,
        λ_values, 
        adap_data, 
        name="Adaptive",
        marker='◆'  # Diamond character
    )
    
    return p
end

# Metrics to plot
plot_configs = [
    ("Feasibility", sequential_feasibility, adaptive_feasibility),
    ("Optimality", sequential_optimality, adaptive_optimality),
    ("Matrix-Vector Products", sequential_matvecs, adaptive_matvecs),
    ("Computation Time (s)", sequential_time, adaptive_time)
]

# Generate and print all plots
for (metric_name, seq_data, adap_data) in plot_configs
    println("\n\n$metric_name vs λ (log-log plot):")
    plot = create_comparison_plot(λ_values, seq_data, adap_data, metric_name)
    println(plot)
end