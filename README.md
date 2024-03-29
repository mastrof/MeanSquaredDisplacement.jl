# MeanSquaredDisplacement

[![Build Status](https://github.com/mastrof/MeanSquaredDisplacement.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mastrof/MeanSquaredDisplacement.jl/actions/workflows/CI.yml?query=branch%3Amain)

MeanSquaredDisplacement.jl provides Julia functions (`imsd` and `emsd`) to compute the
mean squared displacement (MSD) of timeseries.

`imsd` is used to evaluate the MSD of individual timeseries,
while `emsd` evaluates ensemble averages.

MeanSquaredDisplacement.jl relies on [Autocorrelations.jl](https://github.com/mastrof/Autocorrelations.jl)
to compute MSDs, which uses a plain correlation algorithm (O(N^2)) for short timeseries
and a FFT-based algorithm (O(NlogN)) for larger ones, ensuring good performance
for all sample sizes.

The MSD of non-scalar timeseries can also be evaluated
(equivalent to summing the MSD of each scalar element), but it's not yet optimized.


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

## MSD of uneven timeseries
```julia
using MeanSquaredDisplacement
N = 100
Tmin, Tmax = 100, 2000
# trajectories of random length between Tmin and Tmax
x = [cumsum(randn(rand(Tmin:Tmax))) for _ in 1:N]
iM = imsd.(x)
eM = emsd(x)
```
![Individual and ensemble MSD for brownian motions of random lengths](msd.svg)
