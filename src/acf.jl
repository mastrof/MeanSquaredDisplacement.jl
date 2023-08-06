"""
    acf(x, lags)
Return the autocovariance function of timeseries `x` at lag times `lags`, using
a raw acf calculation when the timeseries is small (less than 1024 samples), or
a FFT-based calculation when the timeseries is large.
"""
@inline acf(x, lags) = size(x,1) < 1024 ? rawacf(x, lags) : fftacf(x, lags)

#== Autocovariance function with FFT ==#
function fftacf(x::AbstractVector{<:Number}, lags=0:size(x,1)-1)
    S = float(eltype(x))
    out = Vector{S}(undef, length(lags))
    fftacf!(out, x, lags)
end
function fftacf!(r::AbstractVector, x::AbstractVector, lags)
    lx = length(x)
    m = length(lags)
    @assert length(r) == m
    check_lags(lx, lags)
    A = conv(x, reverse(x))
    for k in 1:m
        δ = lags[k]
        r[k] = A[δ+lx]/(lx-δ)
    end
    return r
end

function fftacf(x::AbstractVector{T}, lags=0:size(x,1)-1) where {T<:Union{AbstractVector,NTuple}}
    S = float(eltype(eltype(x)))
    out = Vector{S}(undef, length(lags))
    fftacf!(out, x, lags)
end
function fftacf!(r::AbstractVector, x::AbstractVector{T}, lags) where {T<:Union{AbstractVector,NTuple}}
    lx = length(x)
    m = length(lags)
    @assert length(r) == m
    check_lags(lx, lags)
    y = [getindex.(x,i) for i in eachindex(first(x))]
    A = sum([conv(s, reverse(s)) for s in y])
    for k in 1:m
        δ = lags[k]
        r[k] = A[δ+lx]/(lx-δ)
    end
    return r
end

check_lags(lx::Int, lags::AbstractVector) = (
    maximum(lags) < lx ||
    error("lags must be less than the sample length.")
)

#== Autocovariance function with autodot ==#
# We follow the structure of `StatsBase.autocov` but
# 1. allow for non-scalar timeseries
# 2. modify the normalization
function rawacf(x::AbstractVector{<:Number}, lags=0:size(x,1)-1)
    S = float(eltype(x))
    out = Vector{S}(undef, length(lags))
    rawacf!(out, x, lags)
end
function rawacf(x::AbstractVector{T}, lags=0:size(x,1)-1) where {T<:Union{AbstractVector,NTuple}}
    S = float(eltype(eltype(x)))
    out = Vector{S}(undef, length(lags))
    rawacf!(out, x, lags)
end

function rawacf!(r::AbstractVector, x::AbstractVector, lags)
    lx = length(x)
    m = length(lags)
    @assert length(r) == m
    check_lags(lx, lags)
    for k in 1:m
        δ = lags[k]
        r[k] = _autodot(x, lx, δ) / (lx-δ)
    end
    return r
end
