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

##
struct RandomWalk{T}
    p::T
    n::Int64
end

function run_rw(p::T, n) where {T}
    x = zero(T)
    xs = [zero(T)]
    for _ in 2:n
        step = 2 * rand(Bernoulli(p)) - 1
        x = x + step
        xs = vcat(xs, x)
    end
    return xs
end

##
struct ScoreELBO{EntropyEst <: AdvancedVI.AbstractEntropyEstimator} <: AdvancedVI.AbstractVariationalObjective
    entropy  ::EntropyEst
    n_samples::Int
end

ScoreELBO(
    n_samples::Int;
    entropy  ::AdvancedVI.AbstractEntropyEstimator = ClosedFormEntropy()
) = ScoreELBO(entropy, n_samples)

#function estimate_objective(
#    rng::Random.AbstractRNG,
#    obj::ScoreELBO,
#    q,
#    prob;
#    n_samples::Int = obj.n_samples
#)
#    samples, entropy = AdvancedVI.reparam_with_entropy(rng, q, q, n_samples, obj.entropy)
#    logprob_samples = logpdf(q, samples)
#    energy = AdvancedVI.estimate_energy_with_samples(prob, samples)
#    logprob_samples * ForwardDiff.value(energy) + entropy
#end
#
#estimate_objective(obj::ScoreELBO, q, prob; n_samples::Int = obj.n_samples) =
#    estimate_objective(Random.default_rng(), obj, q, prob; n_samples)

function compute_kl_divergence(posterior, prior, n_samples, w)
    x = rand(posterior, n_samples)
    log_p = logpdf(posterior, x)[:]
    log_q = logpdf(prior, x)[:]
    kl_div = mean((log_p - log_q))
    return w * kl_div
end

function estimate_scoreelbo_ad_forward(params′, aux)
    @unpack rng, obj, problem, restructure, q_stop = aux
    q = restructure(params′)
    #samples, entropy = AdvancedVI.reparam_with_entropy(rng, q, q_stop, obj.n_samples, obj.entropy)
    samples = rand(rng, q, obj.n_samples)
    logprob_samples = logpdf(q, samples)
    kl_prior = compute_kl_divergence(q, prior, 1000, 10.0)
    model_loss = 0.0
    for (i, sample) in enumerate(samples)
        p = 10 ^ sample
        p = clamp(p, 0, 1)
        x = run_rw(p, n)
        loss = sum((x - observations) .^ 2) / n^2
        model_loss += ForwardDiff.value(loss) * (1 + logprob_samples[i])
    end
    #energy = AdvancedVI.estimate_energy_with_samples(problem, samples)
    #elbo = -model_loss - kl_prior
    #-elbo
    return -model_loss + kl_prior
end

function AdvancedVI.estimate_gradient!(
    rng   ::Random.AbstractRNG,
    obj   ::ScoreELBO,
    adtype::ADTypes.AbstractADType,
    out   ::DiffResults.MutableDiffResult,
    prob,
    params,
    restructure,
    state,
)
    q_stop = restructure(params)
    aux = (rng=rng, obj=obj, problem=prob, restructure=restructure, q_stop=q_stop)
    AdvancedVI.value_and_gradient!(
        adtype, estimate_scoreelbo_ad_forward, params, aux, out
    )
    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)
    out, nothing, stat
end

##


p = 0.3
n = 100
prior = Normal(0, 1)
observations = run_rw(p, n);


##

d = 1
base_dist = DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))

function getq(θ)
    b = PlanarLayer(d)
    for _ in 1:3
        b = b ∘ PlanarLayer(d)
    end
    _, rc = Flux.destructure(b)
    b = rc(θ)
    return transformed(base_dist, b)
end

q = getq(randn(3 * 4))

q_samples = rand(q, 10000)[:]

fig = pairplot((; q=q_samples), PairPlots.Truth(
        (;
            q = log10(p),
        ),
    ))

##

# ELBO objective with the reparameterization gradient
n_montecarlo = 10
elbo = ScoreELBO(n_montecarlo)
model = RandomWalk(p, n)

# Mean-field Gaussian variational family
d = 1
μ = zeros(d);
L = Diagonal(ones(d));
#q = AdvancedVI.MeanFieldGaussian(μ, L);

# Match support by applying the `model`'s inverse bijector
#b = Bijectors.bijector(model)
#binv          = inverse(b)
#q_transformed = Bijectors.TransformedDistribution(q, binv)


# Run inference
max_iter = 1000;
q, stats, _ = AdvancedVI.optimize(
    model,
    elbo,
    q,
    max_iter;
    adtype    = ADTypes.AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
)

##
elbo_loss = [s.elbo for s in stats]
plot(elbo_loss)

# Evaluate final ELBO with 10^3 Monte Carlo samples
#estimate_objective(elbo, q, model; n_samples=10^4)
##
q_samples = rand(q, 10000)[:]
prior_samples = rand(prior, 10000)

fig = pairplot((; p=q_samples), (; p = prior_samples,), PairPlots.Truth(
        (;
            p = log10(p),
        ),
    ))
