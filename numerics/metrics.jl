# Goal: Given A, b, (scaling?)
# Solve the KLLS problem for Ax=b and compare various metrics

using Optim
using DataFrames
using KLLS
using MATLAB
using Random
using NPZ
using CSV
using Dates

function optimToDF(optimState::Vector)
    df = DataFrame(iter=Int[], dual_obj=Float64[], r=Float64[]) #, Δ=T[], Δₐ_Δₚ=T[], cgits=Int[], cgmsg=String[])
    for i in 1:size(optimState,1)
        log = (optimState[i].iteration, optimState[i].value, optimState[i].g_norm)
        push!(df,log)
    end

    return df
end

function dfToCSV(df::DataFrame,method_name::String,problem_name::String)
    dt = Dates.now() 
    dt = Dates.format(dt, "mmdd_HHMM")
    filename = problem_name * "_" * method_name * "_" * dt * ".csv"

    CSV.write("numerics/outputs/" * filename,df)

end

function metrics(
    # Inputs: KLLS Model, Relative tolerance for dual gradient, maximum iterations
        kl::KLLSModel,
        problem_name::String,
        optTol::Real = 1e-6,
        max_iter::Int = 100)

    ## First method, Solve via KLLS and store in tracer, save to CSV
    soln = KLLS.solve!(kl, atol=optTol, rtol=optTol,max_iter = max_iter,trace=true)
    dfToCSV(soln.tracer,"KLLS",problem_name)

    ## Second method, solve dual via KLLS
    ## Def'n of f and nabla f as expected by Optim
    f(y) = KLLS.dObj!(kl,y)
    grad!(grad_in_place,y) = KLLS.dGrad!(kl,y,grad_in_place)

    z0 = zeros(size(kl.A,1)) # Initialization by default zero vector

    opt_sol =optimize(f,grad!,z0,
            LBFGS(),
            Optim.Options(
                store_trace = true,
                show_trace = false,
                extended_trace = true,
                g_tol = optTol,
                f_tol = 0,
                x_tol = 0
            )
            )

    if(Optim.g_converged(opt_sol) == false)
        # TESTING GRADIENT CONVERGENCE SPECIFICALLY!!
        # other conditions for termination may flag
        dfToCSV(optimToDF(opt_sol.trace),"LBFGS",problem_name * "_FAILED");
    else
        dfToCSV(optimToDF(opt_sol.trace),"LBFGS",problem_name);
    end


end