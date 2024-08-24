module UnicodePlotsExt

using KLLS

import UnicodePlots: histogram
function UnicodePlots.histogram(stat::KLLS.ExecutionStats; kwargs...)
    println("")
    UnicodePlots.histogram(stat.solution; kwargs...)
end

end