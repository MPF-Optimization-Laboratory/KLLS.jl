module KLLS

using LinearAlgebra
using Printf
using UnPack
using DataFrames
import Roots
using JSOSolvers: trunk
using NLPModels
# import NLPModels: NLSMeta, NLPModels, NLPModelMeta, AbstractNLPModel, AbstractNLSModel, NLSCounters, Counters, increment!, neval_jprod, neval_jtprod

export solve!, scale!, regularize!, histogram, maximize!, reset!
export KLLSModel, SSModel

include("logsumexp.jl")
include("model.jl")
include("newtoncg.jl")
include("newtonls.jl")
include("selfscale.jl")
include("precon.jl")
include("utils.jl")

end # module