using BlackBIRDS
using LinearAlgebra
import Distributions
import DifferentiationInterface
import Flux
import Functors
import Bijectors
using PairPlots


##

function make_flow(n_dims, n_layers)
    base = Distributions.MultivariateNormal(zeros(n_dims), Diagonal(ones(n_dims)))
    b = Bijectors.PlanarLayer(n_dims)
    for i in 2:n_layers
        b = b ∘ b
    end
    return base, b
end
base, transform = make_flow(3, 1)

flow = Bijectors.transformed(base, b)

rand(flow, 5)


##




struct Normal{T}
    μ::Vector{T}
    σ::Vector{T}
end
Functors.@functor Normal
a = Normal([0.0], [1.0])
v, f = Flux.destructure(a)

