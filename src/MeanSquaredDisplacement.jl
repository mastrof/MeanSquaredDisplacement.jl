module MeanSquaredDisplacement

using AxisArrays
using LinearAlgebra: dot
using DSP
using OffsetArrays
using StatsBase

export emsd, imsd, unfold!

#== Ensemble MSD ==#
"""
    emsd(x::AbstractVector)
Return the ensemble average of the mean squared displacement of each vector in `x`.
`x` can contain timeseries of different lengths.
"""
function emsd(x::AbstractVector{T}) where {T<:AbstractVector}
    individual_msds = imsd.(x)
    tmax = maximum(length.(x))
    S = eltype(first(individual_msds))
    out = zeros(S, tmax)
    counter = zeros(Int, tmax)
    for i in eachindex(x)
        for t in eachindex(x[i])
            out[t] += individual_msds[i][t]
            counter[t] += 1
        end
    end
    for t in eachindex(out)
        out[t] /= counter[t]
    end
    return out
end

"""
    emsd(x::AbstractMatrix, [lags])
Return the ensemble average of the mean squared displacement of each column
of `x` at lag times `lags`.
If not specified `lags` defaults to `0:size(x,1)-1`.
"""
function emsd(x::AbstractMatrix, lags::AbstractVector{<:Integer}=0:size(x,1)-1)
    vec(mean(mapslices(y -> imsd(y, lags), x, dims=1), dims=2))
end


#== Individual MSD ==#
"""
    imsd(x::AbstractMatrix, [lags])
Return the mean squared displacement of each column of `x` at lag times `lags`.
If not specified `lags` defaults to `0:size(x,1)-1`.
"""
function imsd(x::AbstractMatrix, lags::AbstractVector{<:Integer}=0:size(x,1)-1)
    mapslices(y -> imsd(y, lags), x, dims=1)
end

"""
    imsd(x::AbstractVector, [lags])
Return the mean squared displacement of `x` at lag times `lags`.
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
