struct Tracer
    dObj::Vector{Float64} # dual objective
    pResid::Vector{Float64} # primal residuals
    cgits::Vector{Int} # number of CG iterations
end

# Empty tracer
Tracer() = Tracer([], [], [])

import Base: push!

function push!(t::Tracer, dObj, nrmdGrd, cgits)
    push!(t.dObj, dObj)
    push!(t.pResid, nrmdGrd)
    push!(t.cgits, cgits)
end

import Base: getindex
getindex(t::Tracer, i::Int) = (t.dObj[i], t.pResid[i], t.cgits[i])

import Base: length
length(t::Tracer) = length(t.dObj)

