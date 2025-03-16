module UnicodePlotsExt

using Perspectron
import UnicodePlots

# Extend UnicodePlots.histogram to work with Perspectron.ExecutionStats
function UnicodePlots.histogram(stat::Perspectron.ExecutionStats; kwargs...)
    println("")
    UnicodePlots.histogram(stat.solution; kwargs...)
end

end