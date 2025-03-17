module UnicodePlotsExt

using DualPerspective
using UnicodePlots

# Extend UnicodePlots.histogram to work with DualPerspective.ExecutionStats
function UnicodePlots.histogram(stat::DualPerspective.ExecutionStats; kwargs...)
    if "tracer" in propertynames(stat) && !isnothing(stat.tracer)
        # Get the necessary data from the tracer
        if "norm∇d" in propertynames(stat.tracer)
            data = stat.tracer.norm∇d
            return UnicodePlots.histogram(data; title="Gradient Norm Distribution", xlabel="Gradient Norm", kwargs...)
        end
    end
    return UnicodePlots.histogram([0.0]; title="No tracer data available", kwargs...)
end

end