struct SSModel{T, S, K<:PTModel{T}} <: AbstractNLSModel{T,S}
    kl::K
    meta::NLPModelMeta{T,S}
    nls_meta::NLSMeta{T,S}
    counters::NLSCounters
end

"""
    SSModel(kl::PTModel) -> SSModel

Create a self-scaled model from a Perspectron model.

# Arguments
- `kl::PTModel`: The Perspectron model to wrap

# Description
This model is a container for `kl` and the augmented problem for the self-scaled model 
in the variables (y,t).

The default starting point is `(y0, 1.0)`, where `y0` is the starting point of `kl`.
"""
function SSModel(kl::PTModel{T}) where T
    m = kl.meta.nvar
    y0 = kl.meta.x0
    meta = NLPModelMeta(
        m+1,
        x0 = vcat(y0, one(T)),
        name = "Scaled Simplex Model"
    )
    nls_meta = NLSMeta{T, Vector{T}}(m+1, m+1)
    return SSModel(kl, meta, nls_meta, NLSCounters())
end

function Base.show(io::IO, ss::SSModel)
    println(io, "Self-scaled model")
    show(io, ss.kl)
end

NLPModels.reset!(ss::SSModel) = NLPModels.reset!(ss.kl)
