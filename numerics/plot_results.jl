
using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates
using Plots

include("plot_utils.jl")

## 10/15 comment: This is not the correct version; home device need to push to origin.
## in progress of fixing.

folder = "1017_1448"

files_to_plot = readdir(joinpath(@__DIR__ ,"outputs",folder))

###########################################################
#
# Plot all UEG results from a folder
#
#############################################################
UEG_plot = plot(title = "UEG Problem", xlabel = "Cost (Matrix Mult.)", ylabel = "‖∇f‖")
for filename in files_to_plot
    if(occursin("UEG",filename))
        # KLLS and L-BFGS directly comparable in cost
        if(occursin("KLLS",filename))
            plot_KLLS(filename,folder)
        end
        if(occursin("BFGS",filename))
            plot_BFGS(filename,folder)
        end
        
    end
end

display(UEG_plot)


###########################################################
#
# Plot all ρ-meson results from a folder
#
#############################################################

rho_mes_plot = plot(title = "ρ-meson Problem", xlabel = "Cost (Matrix Mult.)", ylabel = "‖∇f‖")
for filename in files_to_plot
    if(occursin("rho",filename))
        # KLLS and L-BFGS directly comparable in cost
        if(occursin("KLLS",filename))
            plot_KLLS(filename,folder)
        end
        if(occursin("BFGS",filename))
            plot_BFGS(filename,folder)
        end
        
    end
end

display(rho_mes_plot)



###########################################################
#
# Plot all MNIST results from a folder
#
#############################################################

MNIST_plot = plot(title = "MNIST Problem", xlabel = "Cost (Matrix Mult.)", ylabel = "‖∇f‖")
for filename in files_to_plot
    if(occursin("MNIST",filename))
        # KLLS and L-BFGS directly comparable in cost
        if(occursin("KLLS",filename))
            plot_KLLS(filename,folder)
        end
        if(occursin("BFGS",filename))
            plot_BFGS(filename,folder)
        end
        
    end
end

display(MNIST_plot)