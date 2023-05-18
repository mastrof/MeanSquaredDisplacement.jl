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
