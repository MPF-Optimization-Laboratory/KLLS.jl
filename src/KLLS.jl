module KLLS

using LinearAlgebra
using Printf
using UnPack
using DataFrames
using UnicodePlots

export solve!, scale!, regularize!, histogram
export KLLSModel

include("logsumexp.jl")
include("model.jl")
include("newtoncg.jl")
include("newtonls.jl")
include("precon.jl")
include("utils.jl")

end # module