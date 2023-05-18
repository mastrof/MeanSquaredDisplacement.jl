# MeanSquaredDisplacement

[![Build Status](https://github.com/mastrof/MeanSquaredDisplacement.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mastrof/MeanSquaredDisplacement.jl/actions/workflows/CI.yml?query=branch%3Amain)

Fast routines for evaluation of mean squared displacements.

## MSD of a single timeseries
```julia
using MeanSquaredDisplacement
x = cumsum(randn(10000))
imsd(x)
```

## Individual MSD of multiple timeseries
```julia
using MeanSquaredDisplacement
x = cumsum(randn(10000, 100))
imsd(x) # evaluates MSD along columns
```

## Ensemble MSD of uneven timeseries
```julia
using MeanSquaredDisplacement
ntrajectories = 50
tmin, tmax = 100, 200
steps = [-1, +1]
# trajectories of random length between tmin and tmax
x = [cumsum(rand(steps, rand(tmin:tmax))) for _ in 1:ntrajectories]
# random starting time for each trajectory
t₀ = rand(0:50, ntrajectories)

emsd(x, t₀)
```
