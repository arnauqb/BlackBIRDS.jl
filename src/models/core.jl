export StochasticModel

abstract type StochasticModel{L} <: Distributions.ContinuousMultivariateDistribution end

Distributions.rand(model::StochasticModel) = throw("Not implemented")
Distributions.rand(model::StochasticModel, n::Int) = hcat([rand(model) for _ in 1:n]...)