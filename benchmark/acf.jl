using BenchmarkTools
using MeanSquaredDisplacement

## Setup
N = 2 .^ (4:2:18)
b_dummy = @benchmark nothing
braw = [b_dummy for _ in N]
bfft = [b_dummy for _ in N]
bsmart = [b_dummy for _ in N]

#== ## Scalar timeseries ==#
for (i,n) in enumerate(N)
    x = cumsum(randn(n))
    lags = 0:size(x,1)-1
    braw[i] = @benchmark MeanSquaredDisplacement.acf($x, $lags)
    bfft[i] = @benchmark MeanSquaredDisplacement.fftacf($x, $lags)
    bsmart[i] = @benchmark MeanSquaredDisplacement.smartacf($x, $lags)
end

[N braw bfft bsmart]

#== ## Non-scalar timeseries ==#
for (i,n) in enumerate(N)
    x = [Tuple(randn(2)) for _ in 1:n]
    lags = 0:size(x,1)-1
    braw[i] = @benchmark MeanSquaredDisplacement.acf($x, $lags)
    bfft[i] = @benchmark MeanSquaredDisplacement.fftacf($x, $lags)
    bsmart[i] = @benchmark MeanSquaredDisplacement.smartacf($x, $lags)
end

[N braw bfft bsmart]
