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

