using BlackBIRDS
using Distributions
using DistributionsAD

struct TestDist{T, B, L} <: StochasticModel{B, L}
    dist::T
    backend::B
    loss::L
end
Distributions.rand(dist::TestDist) = rand(dist.dist)
true_logpdf(dist::TestDist, x) = logpdf(dist.dist, x)

##
n_kde = 100000
test = TestDist(MvNormal(rand(2), 1.0), AutoForwardDiff(), KDELoss(n_kde, BlackBIRDS.MMDKernel()))
data = rand(test)
true_lp_independent = sum(logpdf(Normal(test.dist.μ[i]), data[i]) for i in 1:length(data))
lp = logpdf(test, data);
true_lp = true_logpdf(test, data)

println("Logpdf: ", lp)
println("True logpdf: ", true_lp)
println("True logpdf independent: ", true_lp_independent)

##
struct TestDist2{T, L} <: MultivariateStochasticModel{L}
    dists::Vector{T}
    loss::L
end
Distributions.rand(dist::TestDist2) = hcat([rand(dist.dists[i]) for i in 1:length(dist.dists)]...)
true_logpdf(dist::TestDist2, x) = sum(logpdf(dist.dists[i], x[:, i]) for i in 1:size(x, 2))

n_kde = 100
#test2 = TestDist2([MvNormal(rand(2), 1.0) for _ in 1:10], KDELoss(n_kde, BlackBIRDS.GaussianKernel("auto")))
test2 = TestDist2([MvNormal(rand(2), 1.0) for _ in 1:10], KDELoss(n_kde, BlackBIRDS.MMDKernel()))
data = rand(2, 10)
lp = logpdf(test2, data)

println("Logpdf: ", lp)
println("True logpdf: ", true_logpdf(test2, data))