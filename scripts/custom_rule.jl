using ChainRulesCore
using Optimisers
using ADTypes, ForwardDiff
using LogDensityProblems
using SimpleUnPack
using LinearAlgebra
using Bijectors
using StochasticAD
using AdvancedVI
using Zygote
using Flux
using Distributions
using DynamicPPL
using Random


##
struct RandomWalk{T} <: Distributions.ContinuousMultivariateDistribution
    n::Int64
    p::Vector{T}
end
Flux.@functor RandomWalk (p, )
rw = RandomWalk(100, [0.25])
function Distributions.rand(rw::RandomWalk{T}) where {T}
    xs = [zero(T)]
    for _ in 2:rw.n
        step = 2 * rand(Bernoulli(rw.p[1])) - 1
        x = xs[end] + step
        push!(xs, x)
    end
    return xs
end
Distributions.rand(rng::AbstractRNG, rw::RandomWalk) = rand(rw)
function value_and_gradient(model, params; n_samples=10)
    st_samples = Matrix{Float64}[]
    function f_aux(x)
        _, rec_f = Flux.destructure(model)
        return rand(rec_f(x))
    end
    for _ in 1:n_samples
        fd = StochasticAD.derivative_estimate(f_aux, params)
        push!(st_samples, hcat(fd...))
    end
    v = rand(rw)
    d = sum(st_samples) / n_samples
    return v, d
end
function ChainRulesCore.rrule(::typeof(rand), d::RandomWalk{T}) where {T}
    v, grad = value_and_gradient(d, d.p)
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        d_tangent = Tangent{RandomWalk{T}}(; n=NoTangent(), p = grad' * y_tangent)
        return rand_tangent, d_tangent
    end
    return rand(d), rand_pullback
end
function Distributions.logpdf(rw::RandomWalk, y::Vector)
    x = rand(rw)
    return -sum((x .- y).^2 / rw.n^2)
end

## check rule
p = [0.25]
v, grad = value_and_gradient(rw, p)
v_zygote, v_pull = Zygote.pullback(rand, rw)

##
@model function ppl_model(data, n)
    log_p ~ Normal(0, 1)
    p = 10 .^ log_p
    p = clamp(p, 0.0, 1.0)
    data ~ RandomWalk(n, [p])
end
y = rand(rw)
m = ppl_model(y, 100)
ℓ = DynamicPPL.LogDensityFunction(m)


###

n_montecarlo = 10
d = LogDensityProblems.dimension(ℓ)
μ = zeros(d)
L = Diagonal(ones(d))
q = AdvancedVI.MeanFieldGaussian(μ, L)
elbo = AdvancedVI.RepGradELBO(n_montecarlo)
q_samples_untrained = rand(q, 10^4)[:]

###

max_iter = 10^3
q, stats, _ = AdvancedVI.optimize(
    ℓ,
    elbo,
    q,
    max_iter;
    adtype = ADTypes.AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
)

##
#
using CairoMakie
using PairPlots

elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

# Pair plot
n_samples = 10^4
samples = rand(q, n_samples)[:]
table1 = (; q = samples)
table2 = (; q = q_samples_untrained)
truths = (; q = log10(0.25))
pairplot(table1, table2, PairPlots.Truth(truths))


##






##
struct InferenceModel{M, T}
    model::M
    prior::Distribution
    y::Vector{T}
end

function LogDensityProblems.logdensity(inf_model::InferenceModel, θ)
    prior_logdensity = logpdf(inf_model.prior, θ)[1]
    p = 10 .^ θ
    p = clamp.(p, 0.0, 1.0)
    x = run(inf_model.model, p)
    model_loss = -sum((x .- inf_model.y).^2 / inf_model.model.n^2)
    return prior_logdensity + model_loss
end

LogDensityProblems.dimension(inf_model::InferenceModel) = 1

function LogDensityProblems.capabilities(::Type{<:InferenceModel})
    LogDensityProblems.LogDensityOrder{0}()
end

##
# ELBO objective with the reparameterization gradient

y = run(rw, [0.25]);
prior = Normal(0, 1)
inf_model = InferenceModel(rw, prior, y);
n_montecarlo = 10
elbo         = AdvancedVI.RepGradELBO(n_montecarlo)

# Mean-field Gaussian variational family
d = LogDensityProblems.dimension(inf_model);
μ = zeros(d);
L = Diagonal(ones(d));
q = AdvancedVI.MeanFieldGaussian(μ, L);
q_samples_untrained = rand(q, 10^4)[:]

# Match support by applying the `model`'s inverse bijector
#b             = Bijectors.bijector(model)
#binv          = inverse(b)
q_transformed = q #Bijectors.TransformedDistribution(q, binv)
# Run inference
max_iter = 10^3

q, stats, _ = AdvancedVI.optimize(
    inf_model,
    elbo,
    q_transformed,
    max_iter;
    adtype    = AutoZygote(),
    optimizer = Optimisers.Adam(1e-3)
)

# Evaluate final ELBO with 10^3 Monte Carlo samples
estimate_objective(elbo, q, model; n_samples=10^4)

using CairoMakie
using PairPlots

elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

# Pair plot
n_samples = 10^4
samples = rand(q, n_samples)[:]
table1 = (; q = samples)
table2 = (; q = q_samples_untrained)
truths = (; q = log10(0.25))
pairplot(table1, table2, PairPlots.Truth(truths))


##


