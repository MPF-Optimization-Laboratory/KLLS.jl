module KLLS

using LinearAlgebra
using Printf
using UnPack
using DataFrames
import Roots
using JSOSolvers: trunk
using NLPModels
using LinearOperators

# NonlinearSolve packages
using NonlinearSolve, LinearSolve
import SciMLBase: ReturnCode

export KLLSModel, SSModel, NewtonEQ, TrunkLS
export solve!, scale!, regularize!, histogram, maximize!, reset!

DEFAULT_PRECISION(T) = (eps(T))^(1/3)

include("logsumexp.jl")
include("model.jl")
include("newtoncg.jl")
include("newtonls.jl")
include("selfscale.jl")
include("optimal-transport.jl")
include("precon.jl")
include("utils.jl")

end
