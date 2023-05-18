module MeanSquaredDisplacement

using AxisArrays
using LinearAlgebra: dot
using DSP
using OffsetArrays
using StatsBase

export imsd, unfold!

#== MSD ==#
"""
    imsd(x::AbstractMatrix, [lags])
Return the time-averaged mean squared displacement of each column
of `x` at lag times `lags`.
"""
function imsd(x::AbstractMatrix, lags::AbstractVector{<:Integer}=0:size(x,1)-1)
    mapslices(y -> imsd(y, lags), x, dims=1)
end

"""
    imsd(x::AbstractVector, [lags])
Return the time-averaged mean squared displacement of `x` at lag times `lags`.
If not specified `lags` defaults to `0:length(x)-1`.
"""
function imsd(x::AbstractVector, lags::AbstractVector{<:Integer}=0:size(x,1)-1)
    l = length(x)
    S₂ = acf(x, lags)
    D = OffsetArray([0.0; dot.(x,x); 0.0], -1:l)
    Q = 2*sum(D)
    S₁ = AxisArray(similar(S₂), lags)
    for k in 0:lags[end]
        Q -= (D[k-1] + D[l-k])
        if k ∈ lags
            S₁[atvalue(k)] = Q / (l-k)
        end
    end
    @. S₁ - 2S₂
end


include("acf.jl")
include("unfold.jl")

end # module MeanSquaredDisplacement
