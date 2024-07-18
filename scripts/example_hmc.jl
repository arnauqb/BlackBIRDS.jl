##
using AdvancedHMC
using Random
using Bijectors
using Flux
using DynamicPPL
using DistributionsAD
using Distributions
using LinearAlgebra
using Optimisers
using Zygote
using ForwardDiff
using LogDensityProblems
using LinearAlgebra

using BlackBIRDS
using BlackBIRDS.BrockHommes
using BlackBIRDS.RandomWalk

##

n_timesteps = 100
time_horizon = 1
g2, g3, b2, b3 = 0.9, 0.9, 0.2, -0.2
p = [g2, g3, b2, b3]

abm_model = BrockHommesModel(n_timesteps, p, LLLoss(), time_horizon);
data = rand(abm_model);

lines(data)

Zygote.gradient(x -> sum(rand(x)), abm_model)

##

@model function ppl_model(data, n)
    g2 ~ Uniform(0.0, 1.0)
    g3 ~ Uniform(0.0, 1.0)
    b2 ~ Uniform(0.0, 1.0)
    b3 ~ Uniform(-1.0, 0.0)
    p = [g2, g3, b2, b3]
    data ~ BrockHommesModel(n, p, LLLoss(), time_horizon)
end

model = ppl_model(data, n_timesteps)
ℓπ = DynamicPPL.LogDensityFunction(model)
D = 4
initial_θ = rand(D)
n_samples, n_adapts = 2_000, 1_000

metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)
kernel =  HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(δ, integrator))
nuts = HMCSampler(kernel, metric, adaptor)
samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true)

samples = reduce(hcat, samples)


##
using CairoMakie
using PairPlots

#prior_samples = rand(MvNormal([0.5, 0.5, 0.5, -0.5], 0.25), 10000);
prior_samples = rand(Uniform(-1.0, 1.0), 4, 10000);
function make_table(samples)
    #return (; p = samples[1, :])
    return (;
        g2 = samples[1, :],
        g3 = samples[2, :],
        b2 = samples[3, :],
        b3 = samples[4, :],
    )
end
table = make_table(samples);
table_prior = make_table(prior_samples);
truths = (; g2 = g2, g3 = g3, b2 = b2, b3 = b3);
#truths = (; p = log10(true_p[1]))
pairplot(table, table_prior, PairPlots.Truth(truths, color = "black"))