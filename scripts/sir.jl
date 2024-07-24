using AdvancedVI
using Bijectors
using Distributions
using DistributionsAD
using DifferentiationInterface
using DynamicPPL
using Functors
using Flux
using LinearAlgebra
using Optimisers
using PyPlot
using ProgressMeter
using Zygote

using BlackBIRDS
using DiffABM

##
function make_sir_abm(initial_infected, betas, gamma; backend, discrete_sampler, loss)
    n_agents = 250
    venues = Symbol.(["household", "company", "school", "leisure"])
    fraction_population_per_venue = [1.0, 0.4, 0.4, 1.0]
    number_per_venue = [1, 1, 1, 1] #Int.(floor.([1 / 5, 1 / 10] .* n_agents)) #, 1 / 10, 1 / 2] .* n_agents))
    graph = DiffABM.generate_random_world_graph(
        n_agents, venues, fraction_population_per_venue, number_per_venue)
    n_timesteps = 30
    delta_t = 1.0
    infection_type = DiffABM.ConstantInfection()
    policies = DiffABM.Policies()
    params = DiffABM.SIRParams(
        graph, [initial_infected], betas, venues, [gamma], delta_t, n_timesteps,
        discrete_sampler, infection_type, policies)

    abm = ABM(params, backend, loss)
    return abm
end
true_initial_infected = 0.01
true_gamma = 0.05
true_betas = [0.2, 0.1, 0.1, 0.1]
discrete_sampler = DiffABM.SAD()
loss = MSELoss(10)
#loss = GaussianMMDLoss(10)
#backend = AutoForwardDiff() #AutoStochasticAD(10)
backend = AutoStochasticAD(10)
abm = make_sir_abm(true_initial_infected, true_betas, true_gamma;
    backend = backend, discrete_sampler = discrete_sampler, loss = loss)
@functor DiffABM.SIRParams (venue_betas,)
_, abm_rec = Flux.destructure(abm)
##

data = rand(abm)

fig, ax = subplots()
ax.plot(data)
fig

##
@model function ppl_model(data, abm_rec)
    p ~ filldist(Uniform(0, 1), 4)
    abm = abm_rec(p)
    data ~ abm
end
d = 4
#q = make_masked_affine_autoregressive_flow_torch(d, 4, 16);
μ = -0.4 * ones(d) #randn(d);
L = LowerTriangular(Diagonal(0.25 .* ones(d)));
q = AdvancedVI.FullRankGaussian(μ, L)
#q = make_planar_flow(d, 5)
#q = make_affine_flow(d, 4, 16);
optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(5e-3))
#optimizer = Optimisers.Adam(1e-3)
prob_model = ppl_model(data, abm_rec);
q_samples_untrained = rand(q, 10000);

q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 10,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy()
);

##
elbo_vals = [s.elbo for s in stats];
fig, ax = subplots()
ax.plot(elbo_vals)
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples = rand(q, 10000)
prior_samples = rand(Uniform(0, 1), (4, 10000))
#prior = MvNormal(-0.75 * ones(4), 1.0)
#prior_samples = rand(prior, 10000)
pygtc.plotGTC([q_samples', prior_samples'], #, q_samples_untrained'],
    figureSize = 7, truths = [true_betas...],#[true_initial_infected, true_betas..., true_gamma],
    chainLabels = ["flow", "prior"])

##
pygtc.plotGTC([q_samples', prior_samples],
    figureSize = 7, truths = log.([betas..., gamma]),
    chainLabels = ["flow", "prior"])

##
prior_samples = rand(LogNormal(-2, 0.5), (4, 10000))
pygtc.plotGTC([prior_samples'],
    figureSize = 5,
    chainLabels = ["prior"])

##
pygtc.plotGTC([q_samples'],
    figureSize = 7, truths = log.([betas..., gamma]))
