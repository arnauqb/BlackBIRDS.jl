using Optimisers
using ADTypes, ForwardDiff
using LogDensityProblems
using SimpleUnPack
using LinearAlgebra
using Bijectors
using StochasticAD
using AdvancedVI


##
struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    LogDensityProblems.LogDensityOrder{0}()
end


function Bijectors.bijector(model::NormalLogNormal)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:1+length(μ_y)])
end

##
n_dims = 10
μ_x    = randn()
σ_x    = exp.(randn())
μ_y    = randn(n_dims)
σ_y    = exp.(randn(n_dims))
model  = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y.^2))

##
# ELBO objective with the reparameterization gradient
n_montecarlo = 10
elbo         = AdvancedVI.RepGradELBO(n_montecarlo)

# Mean-field Gaussian variational family
d = LogDensityProblems.dimension(model)
μ = zeros(d)
L = Diagonal(ones(d))
q = AdvancedVI.MeanFieldGaussian(μ, L)

# Match support by applying the `model`'s inverse bijector
b             = Bijectors.bijector(model)
binv          = inverse(b)
q_transformed = Bijectors.TransformedDistribution(q, binv)


# Run inference
max_iter = 10^3
q, stats, _ = AdvancedVI.optimize(
    model,
    elbo,
    q_transformed,
    max_iter;
    adtype    = AutoStochasticAD(10),
    optimizer = Optimisers.Adam(1e-3)
)

# Evaluate final ELBO with 10^3 Monte Carlo samples
estimate_objective(elbo, q, model; n_samples=10^4)