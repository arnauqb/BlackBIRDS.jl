using AdvancedVI
using Bijectors
using Distributions
using DistributionsAD
using DiffSIR
using DifferentiationInterface
using DynamicPPL
using Functors
using Flux
using LinearAlgebra
using Optimisers
using PyPlot
using Zygote

using BlackBIRDS
using BlackBIRDS.SIRJune

##
n_agents = 1000
venues = ["household", "company", "school", "leisure"]
fraction_population_per_venue = [1.0, 0.4, 0.4, 1.0]
number_per_venue = Int.(floor.([1 / 5, 1 / 10, 1 / 10, 1 / 2] .* n_agents))
graph = generate_random_world_graph(
    n_agents, venues, fraction_population_per_venue, number_per_venue)
true_initial_infected = 0.01
true_gamma = 0.05
true_betas = [0.15, 0.2, 0.3, 0.1]
n_timesteps = 30
#loss = KDELoss(20, BlackBIRDS.MMDKernel())
loss = MSELoss(1.0)
discrete_sampler = DiffSIR.SAD()
sir = SIRJuneModel(
    graph, true_initial_infected, true_betas, true_gamma, n_timesteps, discrete_sampler, loss);
@functor SIRJuneModel (venue_betas,)

##

data = rand(sir)

fig, ax = subplots()
ax.plot(data)
fig

##
#loss = KDELoss(10) #
loss = GaussianMMDLoss(data, 1e-3)
@model function ppl_model(
        data, true_initial_infected, true_gamma, n_timesteps, graph, discrete_sampler, loss)
    betas ~ filldist(LogNormal(-1, 0.5), 4)
    data ~ SIRJuneModel(
        graph, true_initial_infected, betas, true_gamma, n_timesteps, discrete_sampler, loss)
end
d = 4
#q = make_masked_affine_autoregressive_flow_torch(d, 8, 32);
q = make_planar_flow(d, 5)
#q = make_affine_flow(d, 4, 16);
#optimizer = Optimisers.OptimiserChain(Optimisers.AdamW(1e-3), Optimisers.ClipNorm(1.0))
optimizer = Optimisers.AdamW(1e-3)
prob_model = ppl_model(
    data, true_initial_infected, true_gamma, n_timesteps, graph, discrete_sampler, loss);
#q_samples_untrained = rand(q, 10000);

q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 100,
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
prior_samples = rand(LogNormal(-1, 0.5), (4, 10000))
pygtc.plotGTC([log.(q_samples'), log.(prior_samples')],
    figureSize = 7, truths = log.([true_betas...]),
    chainLabels = ["flow", "prior"])

##
pygtc.plotGTC([q_samples', prior_samples],
    figureSize = 7, truths = log.([betas..., gamma]),
    chainLabels = ["flow", "prior"])

##
prior_samples = rand(LogNormal(-2, 0.5), (2, 10000))
pygtc.plotGTC([prior_samples'],
    figureSize = 5, truths = [betas..., gamma],
    chainLabels = ["prior"])

##
pygtc.plotGTC([q_samples'],
    figureSize = 7, truths = log.([betas..., gamma]))
