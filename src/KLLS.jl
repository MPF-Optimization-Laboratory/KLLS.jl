module KLLS

using LinearAlgebra
using Printf
using UnPack
using DataFrames
import Roots
using JSOSolvers: trunk
using NLPModels
# using NonlinearSolve, LinearSolve

export KLLSModel, SSModel
export solve!, scale!, regularize!, histogram, maximize!, reset!

DEFAULT_PRECISION(T) = (eps(T))^(1/3)

include("logsumexp.jl")
include("model.jl")
include("newtoncg.jl")
include("newtonls.jl")
include("selfscale.jl")
include("nlsolve.jl")
include("precon.jl")
include("utils.jl")

end
