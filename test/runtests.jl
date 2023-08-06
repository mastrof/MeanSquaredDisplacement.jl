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
    using Statistics: mean
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

    @testset "Unfolding" begin
        function wrap(x,L)
            s = x
            while !(0 ≤ s < L)
                δ = s < 0 ? +L : s > L ? -L : 0
                s += δ
            end
            return s
        end
        x = cumsum(rand(1000))
        # wrap trajectory between 0 and L
        L = 10
        y = wrap.(x, L)
        z = copy(y)
        unfold!(y, L)
        @test y ≈ x

        Lx, Ly = 8.6, 13.0
        x = (1:100) .% Lx
        y = (1:100) .% Ly
        traj = Tuple.(zip(x,y))
        utraj = copy(traj)
        unfold!(utraj, (Lx, Ly))
        @test first.(utraj) == (1:100)
        @test last.(utraj) == (1:100)
    end
end
