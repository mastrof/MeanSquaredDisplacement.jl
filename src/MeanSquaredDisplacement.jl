module MeanSquaredDisplacement

using AxisArrays
using LinearAlgebra: dot
using DSP
using OffsetArrays
using StatsBase

export msd, unfold!

#== MSD ==#
"""
    msd(x::AbstractMatrix, [lags])
Return the mean squared displacement of each column of `x` at lag times `lags`.
"""
function StatsBase.msd(x::AbstractMatrix, lags::AbstractVector{<:Integer}=0:size(x,1)-1)
    mapslices(y -> msd(y, lags), x, dims=1)
end

"""
    msd(x::AbstractVector, [lags])
Return the mean squared displacement of `x` at lag times `lags`.
If not specified `lags` defaults to `0:length(x)-1`.
"""
function StatsBase.msd(x::AbstractVector, lags::AbstractVector{<:Integer}=0:size(x,1)-1)
    l = length(x)
    S₂ = smartacf(x, lags)
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

"""
    smartacf(x, lags)
Return the autocovariance function of timeseries `x` at lag times `lags`, using
a raw acf calculation when the timeseries is small (less than 1024 samples), or
a FFT-based calculation when the timeseries is large.
"""
@inline smartacf(x, lags) = size(x,1) < 1024 ? acf(x, lags) : fftacf(x, lags)

#== Autocovariance function with FFT ==#
function fftacf(x::AbstractVector{<:Tuple}, lags=0:size(x,1)-1)
    T = float(eltype(eltype(x)))
    l = size(x,1)
    y = [getindex.(x,i) for i in eachindex(first(x))]
    A = sum([conv(s, reverse(s)) for s in y])
    [T(A[k+l]/(l-k)) for k in lags]
end
function fftacf(x::AbstractVector{<:Number}, lags=0:size(x,1)-1)
    T = float(eltype(x))
    l = size(x,1)
    A = conv(x, reverse(x))
    [T(A[k+l]/(l-k)) for k in lags]
end

#== Autocovariance function ==#
# We follow the structure of `StatsBase.autocov` but
# 1. allow for non-scalar timeseries
# 2. modify the normalization
"""
    acf(x, [lags])
Return the autocovariance function of timeseries `x` at lag times `lags`.
"""
function acf(x::AbstractVector{<:Number}, lags::AbstractVector{<:Integer})
    S = float(eltype(x))
    out = Vector{S}(undef, length(lags))
    acf!(out, x, lags)
end
function acf(x::AbstractVector{T}, lags::AbstractVector{<:Integer}) where {T<:Union{AbstractVector,NTuple}}
    S = float(eltype(eltype(x)))
    out = Vector{S}(undef, length(lags))
    acf!(out, x, lags)
end

function acf!(r::AbstractVector, x::AbstractVector, lags::AbstractVector{<:Integer})
    lx = length(x)
    m = length(lags)
    @assert length(r) == m
    StatsBase.check_lags(lx, lags)
    for k in 1:m
        r[k] = StatsBase._autodot(x, lx, lags[k]) / (lx-lags[k])
    end
    return r
end

# Extend to accept non-scalar timeseries
StatsBase._autodot(x::AbstractVector{T}, lx::Int, l::Int) where {T<:Union{AbstractVector,NTuple}} =
    dot(view(x, 1:(lx-l)), view(x, (1+l):lx))


#== Unfolding ==#
"""
    unfold!(x::AbstractVector, P)
Unfold timeseries `x` from a domain of periodicity `P`.
"""
function unfold!(x::AbstractVector, P::Real)
    indices = eachindex(x)
    ind_prev = @view indices[1:end-1]
    ind_next = @view indices[2:end]
    for (i,j) in zip(ind_prev, ind_next)
        xₚ = x[i]
        xₙ = x[j]
        x[j] = unfold(xₙ, xₚ, P)
    end
    return x
end

NTupleOrVec = Union{NTuple{D,<:Real},AbstractVector{<:Real}} where D
function unfold(xₙ::AbstractVector, xₚ::AbstractVector, P::Real)
    @assert length(xₙ) == length(xₚ)
    map(i -> unfold(xₙ[i], xₚ[i], P), eachindex(xₙ))
end
function unfold(xₙ::NTuple{D}, xₚ::NTuple{D}, P::Real) where D
    ntuple(i -> unfold(xₙ[i], xₚ[i], P), D)
end
function unfold(xₙ::AbstractVector, xₚ::AbstractVector, P::NTupleOrVec)
    @assert length(xₙ) == length(xₚ) == length(P)
    map(i -> unfold(xₙ[i], xₚ[i], P[i]), eachindex(xₙ))
end
function unfold(xₙ::NTuple{D}, xₚ::NTuple{D}, P::NTupleOrVec) where D
    @assert D == length(P)
    ntuple(i -> unfold(xₙ[i], xₚ[i], P[i]), D)
end

"""
    unfold(xₙ::Real, xₚ::Real, P::Real)
Unfold the value of `xₙ` with respect to `xₚ` from a
domain of periodicity `P`.
"""
function unfold(xₙ::Real, xₚ::Real, P::Real)
    Δx = xₙ - xₚ
    a = round(abs(Δx/P))
    if abs(Δx) > P/2
        return xₙ - a*P*sign(Δx)
    else
        return xₙ
    end
end

end # module MeanSquaredDisplacement
