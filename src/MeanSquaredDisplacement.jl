module MeanSquaredDisplacement

using AxisArrays
using LinearAlgebra: dot
using DSP
using OffsetArrays
using StatsBase

export emsd, imsd, unfold!

#== Ensemble MSD ==#
"""
    emsd(x::AbstractVector, t₀::AbstractVector)
Return the ensemble average of the mean squared displacement of each vector in `x`.
The timeseries in `x` can have different lengths, and different starting times `t₀`.
"""
function emsd(x::AbstractVector{T}, t₀::AbstractVector{<:Integer}) where {T<:AbstractVector}
    @assert length(x) == length(t₀)
    individual_msds = imsd.(x)
    indices = [t₀[i] .+ eachindex(individual_msds[i]) .- 1 for i in eachindex(t₀)]
    t₁, t₂ = extrema(vcat(indices...))
    S = eltype(first(individual_msds))
    out = AxisArray(zeros(S, t₂-t₁+1), t=t₁:t₂)
    counter = AxisArray(zeros(Int, t₂-t₁+1), t₁:t₂)
    for i in eachindex(t₀)
        for j in eachindex(x[i])
            t = t₀[i]+j-1
            out[atvalue(t)] += individual_msds[i][j]
            counter[atvalue(t)] += 1
        end
    end
    for t in t₁:t₂
        out[atvalue(t)] /= counter[atvalue(t)]
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
