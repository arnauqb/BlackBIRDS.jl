using Test
using Distributions
using BlackBIRDS
using LinearAlgebra

struct TestModel{L} <: StochasticModel{L}
    loss::L
end

@testset "test MSELoss" begin
    Distributions.rand(::TestModel) = 3 .* ones(5)
    mseloss = MSELoss(4.0)
    d = TestModel(mseloss)

    y = ones(5)
    @test logpdf(d, y) == -1.0

    Distributions.rand(::TestModel) = hcat(2 .* ones(5), 3 .* ones(5))
    y = ones(5, 2)
    @test logpdf(d, y) == -5 / 4
end


@testset "test KDELoss" begin
    Distributions.rand(::TestModel) = rand(Normal(2, 3), 1)
    kdeloss = KDELoss(2000, Normal(), "auto")
    d = TestModel(kdeloss)

    y = rand(1)
    @test logpdf(d, y) ≈ logpdf(Normal(2, 3), y)[1] rtol = 0.05
end

@testset "test MMD loss" begin
    y = reshape([1., 4, -2], 1, 3)
    mmd = GaussianMMDLoss(y, 1.0)
    @test mmd.sigma == 3.0
    kernel_yy = exp.(-[[0,9,9] [9, 0, 36.] [9, 36, 0]] / (18)) 
    kernel_yy = kernel_yy - I(3)
    @test mmd.kernel_yy == kernel_yy

    a = rand(30, 50)
    Distributions.rand(::TestModel) = a
    mmd = GaussianMMDLoss(a, 1.0)
    d = TestModel(mmd)

    @test logpdf(d, a) ≈ 0.0 rtol = 0.05 atol = 0.05
end
