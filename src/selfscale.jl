struct SSModel{T, S, K<:KLLSModel{T}} <: AbstractNLSModel{T,S}
    kl::K
    meta::NLPModelMeta{T,S}
    nls_meta::NLSMeta{T,S}
    counters::NLSCounters
end

"""
Constructor swallows a `KLLSModel` and returns an `SSModel`.
This new model has one more variable than the original model: `m+1`.
Default starting point is `(zeros(m), 1)`.
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

"""
    residual!(ss, yt, Fx)

Compute the residual in the self-scaling optimality conditions augmented problem, which concatenate the dual residual with the optimal scaling condition:

    F(y, t) = [ ∇d(y)
                logexp(A'y) - log(t) - 1 ]

where `f(y) = logΣexp(A'y)`.
""" 
function NLPModels.residual!(ss::SSModel, yt, Fx)
	increment!(ss, :neval_residual)
    kl = ss.kl
	@unpack A, c, lse = kl
	m = kl.meta.nvar
    r = @view Fx[1:m]
	y = @view yt[1:m]
	t =       yt[end]
    
    scale!(kl, t)             # Apply the latest scaling factor
    f = lseatyc!(kl, y)       # f = logΣexp(A'y). Needed before grad eval
	dGrad!(kl, y, r)          # r ≡ Fx[1:m] = ∇d(y)	
	Fx[end] = f - log(t) - 1  # optimal scaling condition
	return Fx
end

"""
    Jyt = jprod_residual!(ss, yt, zα, Jyt)

Compute the Jacobian-vector product, 

    (1) [ ∇²d(A'y)  Ax  ][ z ] := [ Jy ]  where x:=x(y)
    (2) [ (Ax)'     -1/t][ α ] := [ Jt ]
"""
function NLPModels.jprod_residual!(ss::SSModel, yt, zα, Jyt)

    kl = ss.kl
    @unpack A, lse, mbuf = kl
    Ax = mbuf
    m = kl.meta.nvar
    x = grad(lse)

    increment!(ss, :neval_jprod_residual)

    Jy = @view Jyt[1:m]
    t = yt[end]
    z = @view zα[1:m]
    α = zα[end]
   
    mul!(Ax, A, x)

    # Equation (1)
    dHess_prod!(kl, z, Jy)  # Jy = ∇²d(A'y)z
    Jy .+= α*Ax             # Jy += αAx
    
    # Equation (2)  
    Jyt[end] = z⋅Ax - α/t

    return Jyt
end

function NLPModels.jtprod_residual!(ss::SSModel, yt, wα, Jyt)
    increment!(ss, :neval_jtprod_residual)
    NLPModels.jprod_residual!(ss, yt, wα, Jyt)
end


function solve!(ss::SSModel{T}; kwargs...) where T
    trunk_stats = trunk(ss; kwargs...)

    kl = ss.kl
    x = kl.scale.*grad(kl.lse)
    y = trunk_stats.solution[1:end-1]
    t = trunk_stats.solution[end]
    stats = ExecutionStats(
        trunk_stats.status,
        trunk_stats.elapsed_time,       # elapsed time
        trunk_stats.iter,               # number of iterations
        neval_jprod(kl),                # number of products with A
        neval_jtprod(kl),               # number of products with A'
        zero(T),                        # TODO: primal objective
        trunk_stats.objective,          # dual objective
        x,                              # primal solultion `x`
        (kl.λ)*y,                       # residual r = λy
        trunk_stats.dual_feas,          # norm of the gradient of the dual objective
        DataFrame()                     # TODO: tracer 
    )
    return stats
end