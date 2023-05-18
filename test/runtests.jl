using MeanSquaredDisplacement
using Test

@testset "MeanSquaredDisplacement.jl" begin
    x = collect(1:10)
    m = msd(x)
    @test length(m) == length(x)
    @test m ≈ (eachindex(x) .-1 ).^2
    lags = 2:2:6
    mlags = msd(x, lags)
    @test mlags ≈ m[lags.+1]

    x = [(i,0) for i in 1:10]
    m2 = msd(x)
    @test m2 ≈ m

    x = [(i,i) for i in 1:10]
    m3 = msd(x)
    @test m3 ≈ 2m

    x = cumsum(randn(100,3), dims=1)
    m = msd(x)
    m2 = hcat([msd(x[:,i]) for i in axes(x,2)]...)
    @test m ≈ m2
end
