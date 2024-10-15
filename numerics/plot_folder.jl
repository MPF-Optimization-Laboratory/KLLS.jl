
using Optim
using DataFrames
using KLLS
using MATLAB
using Random
using NPZ
using CSV
using Dates
using Plots

## 10/15 comment: This is not the correct version; home device need to push to origin.
## in progress of fixing.

FOLDER = "1015_1158"

files_to_plot = readdir(joinpath(@__DIR__ ,"outputs",FOLDER))
#print(files_to_plot)

for filename in files_to_plot
    UEG_plot = plot!(title = "UEG Residual", xlabel = "Iteration", ylabel = "r")
    if(occursin("UEG",filename))
        df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
        plot!(df[:,1],df[:,3], labels=[split(filename,"lam")[2]], legend=:topleft)
    end
    display(UEG_plot)
end