# Experiments with optimal transport
# Not a unit-test.
using GridLayoutBase
using Plots, OptimalTransport, Statistics, Distances

using KLLS

μsupport = νsupport = range(-2, 2; length=100)
C2 = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)
μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)
ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)


Tsk, Tkl, klstats = let
    ϵ = 0.01*median(C2)
    Tsk = sinkhorn(μ, ν, C2, ϵ)
    Tkl, klstats = let
        ot = KLLS.OTModel(μ, ν, C2, ϵ)
        solve!(ot, trace=true)
    end
    Tsk, Tkl, klstats
end;


P1 = heatmap(μsupport, νsupport, Tsk;
title="Sinkhorn", aspect_ratio=:equal, cbar=false,
showaxis=false, grid=false)
P2 = heatmap(μsupport, νsupport, Tkl;
title="KLLS", aspect_ratio=:equal, cbar=false,
showaxis=false, grid=false)
plot(P1, P2)

plot(μsupport, μ; label=raw"$\mu$", size=(600, 400))
plot!(νsupport, ν; label=raw"$\nu$")

ϵ = 0.01*median(C2)
T = sinkhorn(μ, ν, C2, ϵ)
P1 = heatmap(μsupport, νsupport, T;
title="Sinkhorn", aspect_ratio=:equal, cbar=false,
showaxis=false, grid=false)
μsupport = νsupport = range(-2, 2; length=100)
C2 = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)
μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)
ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)

# Generate the data
μsupport = νsupport = range(-2, 2; length=100)
C2 = pairwise(SqEuclidean(), μsupport', νsupport'; dims=2)

μ = normalize!(exp.(-μsupport .^ 2 ./ 0.5^2), 1)
ν = normalize!(νsupport .^ 2 .* exp.(-νsupport .^ 2 ./ 0.5^2), 1)

ϵ = 0.01 * median(C2)
T = sinkhorn(μ, ν, C2, ϵ)

# Create the plots
P_top = plot(μsupport, μ,
             legend=false, xlabel="", ylabel="", xticks=false, framestyle=:none,lw=2)

P_center = heatmap(μsupport, νsupport, T;
                   aspect_ratio=:equal, colorbar=false,
                   showaxis=false, grid=false)

P_right = plot(ν, νsupport,
               legend=false, xlabel="", ylabel="", yticks=false, yflip=true, framestyle=:none,lw=2)

# Define the layout using the grid function
layout = grid(2, 2, heights=[0.2, 0.8], widths=[0.8, 0.2])

# Create an empty plot to fill the unused cell
empty_plot = plot(legend=false, framestyle=:none, grid=false, axis=false)



# Combine the plots with reduced spacing
plot(
    P_top, empty_plot,
    P_center, P_right,
    layout = layout,
    size = (600, 600),
    margin = 0Plots.mm,       # Reduce the outer margin
    spacing = 0Plots.mm       # Reduce spacing between subplots
)







