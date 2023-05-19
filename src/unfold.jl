"""
    unfold!(x::AbstractVector, P)
Unfold timeseries `x` from a domain of periodicity `P`.
"""
function unfold!(x::AbstractVector, P)
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


