###############################################################################
#
# solve_metrics.jl
# Takes as input: KLLS model (A,b,mu, ...) & problem name (string)
# Outputs: Solves problem with:
#           - KKLS algorithm
#           - L-BFGS (julia)
# Saves per iteration infomation (dual objective value, dual gradient value
# as dated CSV file, in numerics/outputs/... directory
#
##############################################################################

using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates

include("Optim.jl")

function optimToDF(optimState::Vector,cumulative_cost::Vector)
    df = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[],f_evals =Int[],∇f_evals = Int[]) #, Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])
    for i in 1:size(optimState,1)
        log = (optimState[i].iteration, optimState[i].value, optimState[i].g_norm,cumulative_cost[i][1],cumulative_cost[i][2])
        push!(df,log)
    end

    return df
end

function dfToCSV(df::DataFrame,method_name::String,problem_name::String,lam_str::String)
    dt = Dates.now() ;
    dt = Dates.format(dt, "mmdd_HHMM");
    filename = problem_name * "_" * method_name * "_lam" *lam_str * ".csv";
    dir_name = joinpath(@__DIR__, "outputs", dt);
        if !ispath(dir_name)
            mkpath(dir_name);
        end

    CSV.write(joinpath(dir_name , filename),df);

end

function solve_metrics(
    # Inputs: KLLS Model, Relative tolerance for dual gradient, maximum iterations
        kl::KLLSModel,
        problem_name::String,
        optTol::Real = 1e-6,
        max_iter::Int = 400)

    ## First method, Solve via KLLS and store in tracer, save to CSV
    print("in solver before KLLS")
    soln = KLLS.solve!(kl, atol=optTol, rtol=optTol,max_iter = max_iter,trace=true)
    # Tracer stores CGITS already, very easy to use
    if(soln.status == :optimal)
        # Using sol'n optimality flag
        dfToCSV(soln.tracer,"KLLS",problem_name,string(kl.λ))
    else
        dfToCSV(soln.tracer,"KLLS",problem_name* "_FAILED",string(kl.λ))
    end
    
    ## Second method, solve dual via KLLS
    ## Def'n of f and nabla f as expected by Optim
    f(y) = KLLS.dObj!(kl,y)
    grad!(grad_in_place,y) = KLLS.dGrad!(kl,y,grad_in_place)

    #Can see cost of these in obj and dobj

    z0 = zeros(size(kl.A,1)) # Initialization by default zero vector
    cost_iter =[]
    cb = tr -> begin
            push!(cost_iter, [tr[end].metadata["f evals"] , tr[end].metadata["∇f evals"]])
            false
        end

    opt_sol =Optim.optimize(f,grad!,z0,
            Optim.LBFGS(),
            Optim.Options(
                store_trace = true,
                show_trace = false,
                extended_trace = true,
                callback = cb,
                # Piggyback trace to return ∇f and f evals per iteration
                g_tol = optTol,
                f_tol = 0,
                x_tol = 0
            )
            )
    
    if(Optim.g_converged(opt_sol) == false)
        # TESTING GRADIENT CONVERGENCE SPECIFICALLY!!
        # other conditions for termination may flag
        dfToCSV(optimToDF(opt_sol.trace,cost_iter),"LBFGS",problem_name * "_FAILED",string(kl.λ));
    else
        dfToCSV(optimToDF(opt_sol.trace,cost_iter),"LBFGS",problem_name,string(kl.λ))
    end


end