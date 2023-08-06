_autodot(x::AbstractVector{<:Union{Float32, Float64}}, lx::Int, l::Int) = dot(x, 1:(lx-l), x, (1+l):lx)
_autodot(x::AbstractVector{<:Real}, lx::Int, l::Int) = dot(view(x, 1:(lx-l)), view(x, (1+l):lx))
_autodot(x::AbstractVector{<:AbstractVector}, lx::Int, l::Int) = dot(view(x, 1:(lx-l)), view(x, (1+l):lx))
_autodot(x::AbstractVector{<:NTuple{N,<:Real}}, lx::Int, l::Int) where N = dot(view(x, 1:(lx-l)), view(x, (1+l):lx))
