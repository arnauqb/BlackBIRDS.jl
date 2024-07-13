using Test
using BlackBIRDS
using Distributions
using StatsBase

@testset "test distribution" begin
    dist = MultivariateNormal([1, 2, 3.0], 1)
    samples = BlackBIRDS.sample(dist, 10000)
    @test size(samples) == (3, 10000)
    @test mean(samples, dims = 2)≈[1.0, 2.0, 3.0] atol=0.1
    @test var(samples, dims = 2)≈[1.0, 1.0, 1.0] atol=0.1

    # test gradient
    function f(x)
        return rand(Normal(x[1], x[2]))
    end
    value, gradient = BlackBIRDS.value_and_gradient(f, AutoForwardDiff(), [2.0, 1.0])
    @test gradient[1] == 1.0
    @test gradient[2]≈(value - 2.0) / 1.0 atol=0.1
end
