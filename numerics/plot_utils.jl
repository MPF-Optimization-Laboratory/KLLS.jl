# plot_utils.jl

plot_KLLS = function(filename::String, folder::String)
    # KLLS Cost = Cg iterations 
    # each CG iteration  = 2x matrix mult. each iteration
    # tracks cg iter per outer iter, need to cumsum.
    # KLLS error criterion = ‖ ∇f ‖
    df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",folder,filename)))
    plot!(cumsum(df[:,6]),df[:,3], labels="KLLS λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topright)
end

plot_BFGS = function(filename::String,folder::String)
    # BFGS cost = Function & gradient evaluations
    # each f & ∇f eval = 1 matrix mult each.
    # BFGS tracer already does cumsum of cost.
    df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",folder,filename)))
    plot!(df[1:end,4]+df[1:end,5],df[1:end,3], labels="BFGS λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topright)
end

plot_PDCO = function(filename::String,folder::String)
    df = DataFrame(CSV.File(joinpath(@__DIR__ ,"outputs",FOLDER,filename)))
    plot!(cumsum(df[:,5]),df[:,3], labels="PDCO λ="* split(split(filename,"lam")[2],".csv")[1], legend=:topright)
end