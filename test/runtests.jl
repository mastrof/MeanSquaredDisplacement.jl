using MeanSquaredDisplacement
using Test

@testset "MeanSquaredDisplacement.jl" begin
    # ballistic trajectory
    x = collect(1:10)
    m = imsd(x)
    @test length(m) == length(x)
    @test m ≈ (eachindex(x) .-1 ).^2
    lags = 2:2:6
    mlags = imsd(x, lags)
    @test mlags ≈ m[lags.+1]

    x = collect(1:10_000)
    @test imsd(x) ≈ (eachindex(x) .- 1).^2

    # non-scalar msd is the sum of each dimension's msd
    x = [(i,0) for i in 1:10]
    m2 = imsd(x)
    @test m2 ≈ m

    x = [(i,i) for i in 1:10]
    m3 = imsd(x)
    @test m3 ≈ 2m

    # imsd over matrix is hcat of each column's imsd
    x = cumsum(randn(100,3), dims=1)
    m = imsd(x)
    m2 = hcat([imsd(x[:,i]) for i in axes(x,2)]...)
    @test m ≈ m2

    # emsd is the average of imsds
    using StatsBase: mean
    @test emsd(x) ≈ vec(mean(m2, dims=2))

    x = [collect(1:10), collect(1:10)]
    @test emsd(x) ≈ imsd(x[1])

    na, nb = 10, 5
    a = randn(na)
    b = randn(nb)
    x = [a, b]
    m = emsd(x)
    ma = imsd(a)
    mb = [imsd(b); repeat([0.0], na-nb)]
    divisor = [repeat([2], nb); repeat([1], na-nb)]
    @test m ≈ (ma .+ mb) ./ divisor
end
