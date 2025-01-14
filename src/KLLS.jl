module KLLS

using LinearAlgebra
using Printf
using UnPack
using DataFrames
import Roots
using JSOSolvers: trunk, TrunkSolver
using NLPModels
using LinearOperators
using SolverCore

# NonlinearSolve packages
using NonlinearSolve, LinearSolve
import SciMLBase: ReturnCode

export KLLSModel, SSModel, OTModel, LPModel
export NewtonEQ, SSTrunkLS, SequentialSolve, LevelSet, AdaptiveLevelSet
export solve!, scale!, regularize!, histogram, maximize!, reset!, update_y0!

DEFAULT_PRECISION(T) = (eps(T))^(1/3)

include("logsumexp.jl")
include("klls-model.jl")
include("ss-model.jl")
include("newtoncg.jl")
include("newtonls.jl")
include("selfscale.jl")
include("sequential-scale.jl")
include("level-set.jl")
include("optimal-transport.jl")
include("precon.jl")
include("linear-programming.jl")
include("utils.jl")

end
