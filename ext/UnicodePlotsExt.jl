module UnicodePlotsExt

using KLLS
import UnicodePlots

# Extend UnicodePlots.histogram to work with KLLS.ExecutionStats
function UnicodePlots.histogram(stat::KLLS.ExecutionStats; kwargs...)
    println("")
    UnicodePlots.histogram(stat.solution; kwargs...)
end

end