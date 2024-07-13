using Distributions
using Turing
using Random
using PyPlot
using StochasticAD
using ForwardDiff
#using PairPlots
#using CairoMakie
using PyCall
pygtc = pyimport("pygtc")

##

struct RW{T} <: ContinuousMultivariateDistribution
    p::T
    n_timesteps::Int
end

function Distributions.rand(rng::AbstractRNG, rw::RW{T}) where {T}
    #x = zero(T)
    #xs = [zero(T)]
    #for _ in 2:rw.n_timesteps
    #    x += rand(rng, Normal(0, rw.sigma))
    #    push!(xs, x)
    #end
    #return xs
    x = zero(T)
    xs = [zero(T)]
    for _ in 2:(rw.n_timesteps)
        step = 2 * rand(Bernoulli(rw.p)) - 1
        x = x + step
        push!(xs, x)
    end
    return ForwardDiff.value.(xs)
end

function Distributions.logpdf(rw::RW{T}, y::AbstractVector) where {T}
    loss = 0.0
    n_samples = 10
    for _ in 1:n_samples
        x = rand(rw) 
        loss += sum((x - y) .^ 2) / rw.n_timesteps^2
    end
    # score surrogate objective
    return ForwardDiff.value(-loss / n_samples)
end

##

rw = RW(0.3, 100)
x = rand(rw)
fig, ax = subplots()
ax.plot(x)
fig

## HMC

function custom_likelihood(p, x)

end

@model function fit_rw(y)
    log_theta ~ Normal(0, 1)
    println(DynamicPPL.VarInfo(log_theta).logp)
    theta = 10 .^ log_theta
    theta = clamp(theta, 0, 1)
    y ~ RW(theta, 100)
end

#cond_model = fit_rw(data)
#
#vi_orig = DynamicPPL.VarInfo(cond_model)
#spl = DynamicPPL.SampleFromPrior()
#vi_current = DynamicPPL.VarInfo(DynamicPPL.VarInfo(cond_model), spl, vi_orig[spl])
#
#Turing.LogDensityFunction(vi_current, cond_model, spl, DynamicPPL.DefaultContext())


data = rand(RW(0.3, 100))
iterations = 1
ϵ = 0.05
τ = 10

Turing.setadbackend(:forwarddiff)
chain = sample(fit_rw(data), HMC(ϵ, τ), iterations)

##

prior = Normal(0, 1)
prior_samples = rand(prior, (10000,1))
post_samples = chain[:log_p].data

f = pygtc.plotGTC(
    [post_samples, prior_samples],
    figureSize=7, 
    truths = [log10(0.3)], 
    paramNames=["p"],
    chainLabels=["Posterior", "Prior"],)

## VI

using Bijectors
using Flux
using Turing: Variational


d = 1
base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))


function getq(θ)
    b = PlanarLayer(d)
    for _ in 1:5
        b = b ∘ PlanarLayer(d)
    end
    _, rc = Flux.destructure(b)
    b = rc(θ)
    return transformed(base_dist, b)
end

q = getq(randn(3 * 6))

advi = ADVI(1, 10)

@model function fit_rw(y)
    log_theta ~ Normal(0, 1)
    println(DynamicPPL.leafcontext(__context__))
    theta = 10 .^ log_theta
    theta = clamp(theta, 0, 1)
    y ~ RW(theta, 100)
end
m = fit_rw(data)


q_vi = vi(m, advi, getq, randn(3*6), optimizer=Flux.ADAMW())

##
prior = Normal(0, 1)
prior_samples = rand(prior, (10000, 1))
hmc_samples = chain[:log_p].data
vi_samples = rand(q_vi, 10000)'

f = pygtc.plotGTC(
    [vi_samples, hmc_samples, prior_samples],
    figureSize=7, 
    truths = [log10(0.3)], 
    paramNames=["p"],
    chainLabels=["VI", "HMC", "Prior"],)

