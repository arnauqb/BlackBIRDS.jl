using Test
using Distributions
using BlackBIRDS
using LinearAlgebra

struct TestModel{B, L} <: BlackBIRDS.StochasticModel{B, L}
    backend::B
    loss::L
end

@testset "test MSELoss" begin
    Distributions.rand(::TestModel) = 3 .* ones(1, 5)
    mseloss = MSELoss(w=4.0)
    d = TestModel(AutoForwardDiff(), mseloss)

    y = ones(1, 5)
    @test logpdf(d, y) == -5.0
end

@testset "test KDELoss" begin
    struct TestDist{B, T, L} <: BlackBIRDS.StochasticModel{B, L}
        backend::B
        dist::T
        loss::L
    end
    Distributions.rand(dist::TestDist) = rand(dist.dist)
    true_logpdf(dist::TestDist, x) = logpdf(dist.dist, x)
    n_kde = 10000
    test = TestDist(AutoForwardDiff(), MvNormal(rand(2), 1.0),
        KDELoss(n_kde, BlackBIRDS.GaussianKernel("auto")))
    data = rand(test)
    true_lp_independent = sum(logpdf(Normal(test.dist.μ[i]), data[i])
    for i in 1:length(data))
    lp = logpdf(test, data)
    true_lp = true_logpdf(test, data)
    @test lp≈true_lp rtol=0.05
end