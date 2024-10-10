
using Optim
using DataFrames
using KLLS
using MATLAB
using Random
using NPZ
using CSV
using Dates
using Plots

FOLDER = "1010_1521"

files_to_plot = readdir(joinpath(@__DIR__ ,outputs,FOLDER))
print(files_to_plot)
