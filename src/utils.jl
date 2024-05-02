struct Tracer
    dObj::Vector{Float64} # dual objective
    pResid::Vector{Float64} # primal residuals
end

Tracer() = Tracer(Float64[], Float64[])

import Base: push!

function push!(t::Tracer, dObj, nrmdGrd)
    push!(t.dObj, dObj)
    push!(t.pResid, nrmdGrd)
end

