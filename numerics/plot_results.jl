
using DataFrames
using KLLS
using Random
using NPZ
using CSV
using Dates
using Plots

## 10/15 comment: This is not the correct version; home device need to push to origin.
## in progress of fixing.

FOLDER = "1016_1208"

files_to_plot = readdir(joinpath(@__DIR__ ,"outputs",FOLDER))

UEG_plot = plot(title = "UEG Problem", xlabel = "Cost (Matrix Mult.)", ylabel = "‖∇f‖")

for filename in files_to_plot
    if(occursin("UEG",filename))
        # KLLS and L-BFGS directly comparable in cost

        if(occursin("KLLS",filename))
            # KLLS Cost = Cg iterations 
            # each CG iteration  = 2x matrix mult. each iteration
            # KLLS error criterion = ‖ ∇f ‖
            df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
            plot!(df[:,6],df[:,3], labels="KLLS λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topleft)
        end
        if(occursin("BFGS",filename))
            # BFGS cost = Function & gradient evaluations
            # each f & ∇f eval = 1 matrix mult each.
            df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
            plot!(df[1:end,4]+df[1:end,5],df[1:end,3], labels="BFGS λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topleft)
        end
        
        #=

        ###############################################################
        #
        # PDCO does a ridiculous amount of CG iterations, likely needs to be a seperate plot axis'
        #
        ##################################################################

        if(occursin("pdco",filename))
            df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
            plot!(cumsum(df[:,5]),df[:,3], labels="PDCO λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topleft)
        end
        =#
    end
end

display(UEG_plot)


rho_meson_plot = plot(title = "ρ-meson Problem", xlabel = "Cost (Matrix Mult.)", ylabel = "‖∇f‖")

for filename in files_to_plot
    if(occursin("rho_",filename))
        # KLLS and L-BFGS directly comparable in cost

        if(occursin("KLLS",filename))
            # KLLS Cost = Cg iterations 
            # each CG iteration  = 2x matrix mult. each iteration
            # KLLS error criterion = ‖ ∇f ‖
            df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
            plot!(df[:,6],df[:,3], labels="KLLS λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topleft)
        end
        if(occursin("BFGS",filename))
            # BFGS cost = Function & gradient evaluations
            # each f & ∇f eval = 1 matrix mult each.
            df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
            plot!(df[1:end,4]+df[1:end,5],df[1:end,3], labels="BFGS λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topleft)
        end
        
        #=

        ###############################################################
        #
        # PDCO does a ridiculous amount of CG iterations, likely needs to be a seperate plot axis'
        #
        ##################################################################

        if(occursin("pdco",filename))
            df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
            plot!(cumsum(df[:,5]),df[:,3], labels="PDCO λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topleft)
        end
        =#
    end
end

display(rho_meson_plot)