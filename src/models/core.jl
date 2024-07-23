export UnivariateStochasticModel, MultivariateStochasticModel, StochasticModel

abstract type UnivariateStochasticModel{L} <: Distributions.ContinuousMultivariateDistribution end
abstract type MultivariateStochasticModel{L} <: Distributions.ContinuousMatrixDistribution end
StochasticModel{L} = Union{UnivariateStochasticModel{L}, MultivariateStochasticModel{L}}

#Base.length(model::StochasticModel) = throw("Please implement Base.length for model")
#Base.size(model::StochasticModel) = throw("Please implement Base.size for model")
Distributions.rand(model::StochasticModel) = throw("Please implemented the rand method")
Distributions.rand(model::StochasticModel, n::Int) = hcat([rand(model) for _ in 1:n]...)
Distributions.rand(rng::Random.AbstractRNG, model::StochasticModel) = rand(model)