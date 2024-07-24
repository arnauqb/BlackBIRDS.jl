using Test
using Distributions
using BlackBIRDS
using LinearAlgebra

struct TestModel{L} <: StochasticModel{B, L}
    loss::L
end

@testset "test MSELoss" begin
    Distributions.rand(::TestModel) = 3 .* ones(5)
    mseloss = MSELoss(4.0)
    d = TestModel(mseloss)

    y = ones(5)
    @test logpdf(d, y) == -1.0

    Distributions.rand(::TestModel) = hcat(2 .* ones(5), 3 .* ones(5))'
    y = ones(2, 5)
    @test logpdf(d, y) == -5 / 4 / 2
end

@testset "test KDELoss" begin
    struct TestDist{T, L} <: UnivariateStochasticModel{L}
        dist::T
        loss::L
    end
    Distributions.rand(dist::TestDist) = rand(dist.dist)
    true_logpdf(dist::TestDist, x) = logpdf(dist.dist, x)
    n_kde = 10000
    test = TestDist(MvNormal(rand(2), 1.0), KDELoss(n_kde, BlackBIRDS.GaussianKernel("auto")))
    data = rand(test)
    true_lp_independent = sum(logpdf(Normal(test.dist.μ[i]), data[i])
        for i in 1:length(data))
    lp = logpdf(test, data)
    true_lp = true_logpdf(test, data)
    @test lp≈true_lp rtol=0.05
end

@testset "test MMD loss" begin
    y = reshape([1.0, 4, -2], 1, 3)
    mmd = GaussianMMDLoss(y, 1.0)
    @test mmd.sigma == 3.0
    kernel_yy = exp.(-[[0, 9,9] [9, 0,36.0] [9, 36,0]] / (18))
    kernel_yy = kernel_yy - I(3)
    @test mmd.kernel_yy == kernel_yy

    a = rand(30, 50)
    Distributions.rand(::TestModel) = a
    mmd = GaussianMMDLoss(a, 1.0)
    d = TestModel(mmd)

    @test logpdf(d, a)≈0.0 rtol=0.05 atol=0.05
end
