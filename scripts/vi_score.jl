using AdvancedVI
using LogDensityProblems
using SimpleUnPack
using Random
using ADTypes
using DiffResults
using Distributions
using DynamicPPL
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
using DynamicPPL

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
#struct NormalLogNormal{MX, SX, MY, SY}
#    μ_x::MX
#    σ_x::SX
#    μ_y::MY
#    Σ_y::SY
#end
#
#function Distributions.sample(model::NormalLogNormal, n_samples::Int)
#    (; μ_x, σ_x, μ_y, Σ_y) = model
#    x = rand(LogNormal(μ_x, σ_x), n_samples)
#    y = rand(MvNormal(μ_y, Σ_y), n_samples)
#    return hcat(x, y')'
#end
#
#function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
#    (; μ_x, σ_x, μ_y, Σ_y) = model
#    logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
#end
#
#function LogDensityProblems.dimension(model::NormalLogNormal)
#    length(model.μ_y) + 1
#end
#
#function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
#    LogDensityProblems.LogDensityOrder{0}()
#end

#function Bijectors.bijector(model::NormalLogNormal)
#    (; μ_x, σ_x, μ_y, Σ_y) = model
#    Bijectors.Stacked(
#        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
#        [1:1, 2:(1 + length(μ_y))])
#end

##

n_dims = 5
#μ_x = randn()
#σ_x = exp.(randn())
#μ_y = randn(n_dims)
#σ_y = exp.(randn(n_dims))
#model = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y .^ 2))
function wrap_in_vec_reshape(f, in_size)
    vec_in_length = prod(in_size)
    reshape_inner = Bijectors.Reshape((vec_in_length,), in_size)
    out_size = Bijectors.output_size(f, in_size)
    vec_out_length = prod(out_size)
    reshape_outer = Bijectors.Reshape(out_size, (vec_out_length,))
    return reshape_outer ∘ f ∘ reshape_inner
end
function Bijectors.bijector(
    model::DynamicPPL.Model, ::Val{sym2ranges}=Val(false); varinfo=DynamicPPL.VarInfo(model)
) where {sym2ranges}
    num_params = sum([
        size(varinfo.metadata[sym].vals, 1) for sym in keys(varinfo.metadata)
    ])

    dists = vcat([varinfo.metadata[sym].dists for sym in keys(varinfo.metadata)]...)

    num_ranges = sum([
        length(varinfo.metadata[sym].ranges) for sym in keys(varinfo.metadata)
    ])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1

    # ranges might be discontinuous => values are vectors of ranges rather than just ranges
    sym_lookup = Dict{Symbol,Vector{UnitRange{Int}}}()
    for sym in keys(varinfo.metadata)
        sym_lookup[sym] = Vector{UnitRange{Int}}()
        for r in varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            push!(sym_lookup[sym], ranges[range_idx])
            range_idx += 1
        end

        idx += varinfo.metadata[sym].ranges[end][end]
    end

    bs = map(tuple(dists...)) do d
        b = Bijectors.bijector(d)
        if d isa Distributions.UnivariateDistribution
            b
        else
            wrap_in_vec_reshape(b, size(d))
        end
    end

    if sym2ranges
        return (
            Bijectors.Stacked(bs, ranges),
            (; collect(zip(keys(sym_lookup), values(sym_lookup)))...),
        )
    else
        return Bijectors.Stacked(bs, ranges)
    end
end

@model function ppl_model(data)
    μ_x ~ Normal(0, 1)
    σ_x ~ LogNormal(0, 1)
    μ_y ~ MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
    σ_y = zeros(n_dims)
    for i in 1:n_dims
        σ_y[i] ~ LogNormal(0, 1)
    end
    data[1] ~ LogNormal(μ_x, σ_x)
    data[2] ~ MvNormal(μ_y, Diagonal(σ_y .^ 2))
end

m = ppl_model((rand(LogNormal(0.5, 1.0)), randn(n_dims)))
data = rand(m)
ℓ = DynamicPPL.LogDensityFunction(m)
##

# ELBO objective with the reparameterization gradient
n_montecarlo = 50
elbo = AdvancedVI.RepGradELBO(n_montecarlo)
#elbo = ScoreELBO(n_montecarlo)

# Mean-field Gaussian variational family
d = LogDensityProblems.dimension(ℓ)
μ = zeros(d)
L = Diagonal(ones(d))
q = AdvancedVI.MeanFieldGaussian(μ, L)

# Match support by applying the `model`'s inverse bijector
b = Bijectors.bijector(m)
binv = inverse(b)
q_transformed = Bijectors.TransformedDistribution(q, binv)

# Run inference
max_iter = 10^3
q, stats, _ = AdvancedVI.optimize(
    ℓ,
    elbo,
    q_transformed,
    max_iter;
    adtype = ADTypes.AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
)

# Evaluate final ELBO with 10^3 Monte Carlo samples
#estimate_objective(elbo, q, model; n_samples = 10^4)

##
elbo_vals = [s.elbo for s in stats]
plot(elbo_vals)

##
q_samples = rand(q, 10000)
true_samples =  rand(m)

mynames = [Symbol("q_$i") for i in 1:n_dims+1];
myvalues = [q_samples[i,: ] for i in 1:n_dims+1];
table1 = (;zip(mynames, myvalues)...);

mynames = [Symbol("q_$i") for i in 1:n_dims+1];
myvalues = [true_samples[i,: ] for i in 1:n_dims+1];
table2 = (;zip(mynames, myvalues)...);

fig = pairplot(table1, table2)
