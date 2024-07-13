using AdvancedVI
using LogDensityProblems
using SimpleUnPack
using Random
using ADTypes
using DiffResults
using Distributions
using DistributionsAD
using Bijectors
using Flux
using ADTypes, ForwardDiff
using Optimisers
using LinearAlgebra
using StatsBase
using PairPlots
using CairoMakie
using StochasticAD
using Infiltrator

##

struct ScoreELBO{EntropyEst <: AdvancedVI.AbstractEntropyEstimator} <:
       AdvancedVI.AbstractVariationalObjective
    entropy::EntropyEst
    n_samples::Int
end

function ScoreELBO(
        n_samples::Int;
        entropy::AdvancedVI.AbstractEntropyEstimator = ClosedFormEntropy()
)
    ScoreELBO(entropy, n_samples)
end

function estimate_energy_with_samples(prob, samples, samples_logprob)
    a = Base.Fix1(LogDensityProblems.logdensity, prob).(AdvancedVI.eachsample(samples))
    return mean(ForwardDiff.value.(a) .* (samples_logprob .- mean(a)))
end

function compute_elbo(q, samples, entropy, obj, problem)
    samples_logprob = logpdf.(Ref(q), AdvancedVI.eachsample(samples)) 
    energy = estimate_energy_with_samples(problem, samples, samples_logprob)
    elbo = energy + entropy
    return elbo
end

function estimate_scoreelbo_ad_forward(params′, aux)
    @unpack rng, obj, problem, restructure, q_stop = aux
    q = restructure(params′)
    samples, entropy = AdvancedVI.reparam_with_entropy(
        rng, q, q_stop, obj.n_samples, obj.entropy)
    elbo = compute_elbo(q, samples, entropy, obj, problem)
    return -elbo
end

function AdvancedVI.estimate_objective(
        rng::Random.AbstractRNG,
        obj::ScoreELBO,
        q,
        prob;
        n_samples::Int = obj.n_samples
)
    samples, entropy = AdvancedVI.reparam_with_entropy(rng, q, q, n_samples, obj.entropy)
    return compute_elbo(q, samples, entropy, obj, prob)
end

function AdvancedVI.estimate_objective(
        obj::ScoreELBO, q, prob; n_samples::Int = obj.n_samples)
    estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

function AdvancedVI.estimate_gradient!(
        rng::Random.AbstractRNG,
        obj::ScoreELBO,
        adtype::ADTypes.AbstractADType,
        out::DiffResults.MutableDiffResult,
        prob,
        params,
        restructure,
        state
)
    q_stop = restructure(params)
    aux = (rng = rng, obj = obj, problem = prob, restructure = restructure, q_stop = q_stop)
    AdvancedVI.value_and_gradient!(
        adtype, estimate_scoreelbo_ad_forward, params, aux, out
    )
    nelbo = DiffResults.value(out)
    stat = (elbo = -nelbo,)
    out, nothing, stat
end

##
struct NormalLogNormal{MX, SX, MY, SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function Distributions.sample(model::NormalLogNormal, n_samples::Int)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    x = rand(LogNormal(μ_x, σ_x), n_samples)
    y = rand(MvNormal(μ_y, Σ_y), n_samples)
    return hcat(x, y')'
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
        [1:1, 2:(1 + length(μ_y))])
end

##

n_dims = 5
μ_x = randn()
σ_x = exp.(randn())
μ_y = randn(n_dims)
σ_y = exp.(randn(n_dims))
model = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y .^ 2))

##

# ELBO objective with the reparameterization gradient
n_montecarlo = 50
#elbo = AdvancedVI.RepGradELBO(n_montecarlo)
elbo = ScoreELBO(n_montecarlo)

# Mean-field Gaussian variational family
d = LogDensityProblems.dimension(model)
μ = zeros(d)
L = Diagonal(ones(d))
q = AdvancedVI.MeanFieldGaussian(μ, L)

# Match support by applying the `model`'s inverse bijector
b = Bijectors.bijector(model)
binv = inverse(b)
q_transformed = Bijectors.TransformedDistribution(q, binv)

# Run inference
max_iter = 10^3
q, stats, _ = AdvancedVI.optimize(
    model,
    elbo,
    q_transformed,
    max_iter;
    adtype = ADTypes.AutoForwardDiff(),
    optimizer = Optimisers.OptimiserChain(Optimisers.Adam(1e-3), Optimisers.ClipNorm(10.0))
    #optimizer = Optimisers.AdamW(1e-3)
)

# Evaluate final ELBO with 10^3 Monte Carlo samples
#estimate_objective(elbo, q, model; n_samples = 10^4)

##
elbo_vals = [s.elbo for s in stats]
plot(elbo_vals)

##
q_samples = rand(q, 10000)
true_samples =  sample(model, 10000)

mynames = [Symbol("q_$i") for i in 1:n_dims+1];
myvalues = [q_samples[i,: ] for i in 1:n_dims+1];
table1 = (;zip(mynames, myvalues)...);

mynames = [Symbol("q_$i") for i in 1:n_dims+1];
myvalues = [true_samples[i,: ] for i in 1:n_dims+1];
table2 = (;zip(mynames, myvalues)...);

fig = pairplot(table1, table2)