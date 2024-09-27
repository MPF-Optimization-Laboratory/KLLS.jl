function nlresidual!(F, yt, ss::SSModel)
    residual!(ss, yt, F)
end

function nljprod!(Jyt, zα, yt, ss::SSModel)
    jprod_residual!(ss, yt, zα, Jyt)
end


# reset!(kl)
# ff = NonlinearFunction(KLLS.nlresidual!; jvp=KLLS.nljprod!)
# y0 = zeros(m)
# yt0 = vcat(y0, 1.0)
# prob = NonlinearProblem(ff, yt0, ss)
# @suppress_err begin
# sol = solve(
#     prob,
#     reltol=1e-6,
#     abstol=1e-6,
#     show_trace = Val(true),
#     trace_level = TraceAll(),
#     store_trace = Val(true),
#     NewtonRaphson(
#         linsolve = KrylovJL_MINRES(verbose=10, itmax=10),
#        )
#     )
# end;