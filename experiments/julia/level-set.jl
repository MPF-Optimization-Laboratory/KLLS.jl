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

# Set up project environment
import Pkg, TestEnv
script_dir = dirname(@__FILE__)
project_root = abspath(joinpath(script_dir, "..", ".."))
Pkg.activate(project_root)
TestEnv.activate()

# Ensure all required packages are available
using KLLS, LinearAlgebra, UnPack, Printf
import NPZ: npzread
using UnicodePlots  # For Unicode plots

println("Testing level-set algorithms with UEG synthetic test problem...")

# Load test problem data
# Get the absolute path to the project root directory
data_path = joinpath(project_root, "data", "synthetic-UEG_testproblem.npz")
data = npzread(data_path)

# Extract the data
A = data["A"]
b = data["b_avg"]
q = convert(Vector{Float64}, data["mu"])
q .= max.(q, 1e-13)  # Ensure positivity
q .= q./sum(q)       # Normalize to sum to 1
b_std = data["b_std"]
C = inv.(b_std) |> diagm  # Convert standard deviations to weights
c = q  # We'll use the reference distribution as the cost vector

m, n = size(A)

# Set algorithm parameters
maxiter = 1000000
tol = 1e-6
verbose = false

# Define a range of lambda values to test
λ_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
plot_λ_values = reverse(λ_values)  # Reverse for plotting (decreasing from left to right)

"""
    SolverMetrics

Holds performance metrics for a level-set solver.

Fields:
- `mat_vecs::Vector{Float64}`: Number of matrix-vector products
- `feasibility::Vector{Float64}`: Feasibility measure (norm of residual)
- `optimality::Vector{Float64}`: Optimality measure
- `times::Vector{Float64}`: Computation time in seconds
"""
@kwdef struct SolverMetrics
    mat_vecs::Vector{Int64} = Int64[]
    feasibility::Vector{Float64} = Float64[]
    optimality::Vector{Float64} = Float64[]
    times::Vector{Float64} = Float64[]
end

"""
    push!(metrics::SolverMetrics, result)

Add solver results to the metrics collection.
"""
function Base.push!(metrics::SolverMetrics, result)
    push!(metrics.mat_vecs, result.neval_jprod + result.neval_jtprod)
    push!(metrics.feasibility, norm(result.residual, Inf))
    push!(metrics.optimality, result.optimality)
    push!(metrics.times, result.elapsed_time)
end

# Initialize metrics for both methods
seq_metrics = SolverMetrics()
adap_metrics = SolverMetrics()

println("\n# Comparison across multiple λ values\n")

# Test each lambda value
for λ in λ_values
    println("Testing with λ = $λ\n")
    
    # Create KLLS model
    kl_model = KLLSModel(A, b, C=C, q=q, λ=λ)
    
    # Test Sequential Solve method
    seq_result = solve!(kl_model, SequentialSolve(), logging=0)
    push!(seq_metrics, seq_result)
    
    # Create a new model instance for the Adaptive method
    kl_model = KLLSModel(A, b, C=C, q=q, λ=λ)
    
    # Test Adaptive Level-Set method
    adap_result = solve!(kl_model, AdaptiveLevelSet(), logging=0)
    push!(adap_metrics, adap_result)
    
    # Print comparison table
    println("Comparison of Methods (λ = $λ):")
    println("------------------------------------------------------------------------")
    println("Method                   Mat-Vecs  Feasibility   Optimality     Time (s)")
    println("------------------------------------------------------------------------")
    @printf("%-25s %8d    %.2e     %.2e         %.2f\n", 
            "Sequential", seq_metrics.mat_vecs[end], seq_metrics.feasibility[end], 
            seq_metrics.optimality[end], seq_metrics.times[end])
    @printf("%-25s %8d    %.2e     %.2e         %.2f\n", 
            "Adaptive Level-Set", adap_metrics.mat_vecs[end], adap_metrics.feasibility[end], 
            adap_metrics.optimality[end], adap_metrics.times[end])
    println("------------------------------------------------------------------------")
    println()
end

# Create function for generating plots to avoid code duplication
function create_comparison_plot(λ_values, seq_data, adap_data, metric_name; width=80, height=20)
    plot = scatterplot(
        λ_values, seq_data, 
        xlabel="λ", 
        ylabel="$metric_name (log)",
        title="$metric_name vs λ (log-log plot)",
        xscale=:log10,
        yscale=:log10,
        marker='●',
        name="Sequential",
        width=width,
        height=height,
        xflip=true  # Flip x-axis to show decreasing lambda values from left to right
    )
    
    scatterplot!(
        plot,
        λ_values, adap_data,
        marker='◆',
        name="Adaptive"
    )
    
    return plot
end

# Generate log-log plots for different metrics
println("\nFeasibility vs λ (log-log plot):")
feasibility_plot = create_comparison_plot(plot_λ_values, reverse(seq_metrics.feasibility), 
                                        reverse(adap_metrics.feasibility), "Feasibility")
println(feasibility_plot)

println("\nOptimality vs λ (log-log plot):")
optimality_plot = create_comparison_plot(plot_λ_values, reverse(seq_metrics.optimality), 
                                       reverse(adap_metrics.optimality), "Optimality")
println(optimality_plot)

println("\nMatrix-Vector Products vs λ (log-log plot):")
mvp_plot = create_comparison_plot(plot_λ_values, reverse(seq_metrics.mat_vecs), 
                                reverse(adap_metrics.mat_vecs), "Matrix-Vector Products")
println(mvp_plot)

println("\nComputation Time (s) vs λ (log-log plot):")
time_plot = create_comparison_plot(plot_λ_values, reverse(seq_metrics.times), 
                                 reverse(adap_metrics.times), "Computation Time (s)")
println(time_plot)