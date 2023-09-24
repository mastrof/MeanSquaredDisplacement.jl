"""
    unfold(x::AbstractVector, P)
Unfold timeseries `x` from a periodic domain extending from `0` to `P`.
"""
function unfold(x::AbstractVector, P)
    y = deepcopy(x)
    unfold!(y, P)
    return y
end

"""
    unfold!(x::AbstractVector, P)
Unfold timeseries `x` from a periodic domain extending from `0` to `P`.
The periodicity `P` can be either a real number (for a cubic domain) or
a collection (`AbstractVector` or `NTuple`) with one value for each dimension.
"""
function unfold!(x::AbstractVector, P)
    indices = eachindex(x)
    ind_prev = @view indices[1:end-1]
    ind_next = @view indices[2:end]
    for (i,j) in zip(ind_prev, ind_next)
        x0 = x[i]
        x1 = x[j]
        x[j] = unfold(x1, x0, P)
    end
    return x
end

NTupleOrVec = Union{NTuple{D,<:Real},AbstractVector{<:Real}} where D
function unfold(x1::AbstractVector, x0::AbstractVector, P::Real)
    @assert length(x1) == length(x0)
    map(i -> unfold(x1[i], x0[i], P), eachindex(x1))
end
function unfold(x1::NTuple{D}, x0::NTuple{D}, P::Real) where D
    ntuple(i -> unfold(x1[i], x0[i], P), D)
end
function unfold(x1::AbstractVector, x0::AbstractVector, P::NTupleOrVec)
    @assert length(x1) == length(x0) == length(P)
    map(i -> unfold(x1[i], x0[i], P[i]), eachindex(x1))
end
function unfold(x1::NTuple{D}, x0::NTuple{D}, P::NTupleOrVec) where D
    @assert D == length(P)
    ntuple(i -> unfold(x1[i], x0[i], P[i]), D)
end

"""
    unfold(x1::Real, x0::Real, P::Real)
Unfold the value of `x1` with respect to `x0` from a
domain of periodicity `P`.
"""
function unfold(x1::Real, x0::Real, P::Real)
    dx = x1 - x0
    a = round(abs(dx / P))
    if abs(dx) > P/2
        return x1 - a*P*sign(dx)
    else
        return x1
    end
end
