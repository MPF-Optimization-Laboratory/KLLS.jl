struct SSModel{T, S, K<:KLLSModel{T}} <: AbstractNLSModel{T,S}
    kl::K
    meta::NLPModelMeta{T,S}
    nls_meta::NLSMeta{T,S}
    counters::NLSCounters
end

"""
    SSModel(kl::KLLSModel) -> SSModel

This model is a container for `kl` and the augmented problem for the self-scaled model in the variables (y,t).

Default starting point is `(kl.x0, 1.0)`
"""
function SSModel(kl::KLLSModel{T}) where T
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
