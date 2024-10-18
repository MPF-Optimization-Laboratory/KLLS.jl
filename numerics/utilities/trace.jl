# First order methods trace, used by AcceleratedGradientDescent,
# ConjugateGradient, GradientDescent, LBFGS and MomentumGradientDescent
function common_trace!(tr, d, state, iteration, method::FirstOrderOptimizer, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["Current step size"] = state.alpha

        ##########################################################################
        #
        # 10/16/2024 edit: added f and ∇f evaluations to extended tracer dict.
        #
        #######################################################################
        dt["f evals"] = f_calls(d)
        dt["∇f evals"] = g_calls(d)
    end
    g_norm = maximum(abs, gradient(d))
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
