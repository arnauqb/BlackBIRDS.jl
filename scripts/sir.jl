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
using ProgressMeter
using Zygote

using BlackBIRDS
using BlackBIRDS.SIRJune

##
n_agents = 500
venues = ["household", "company", "school", "leisure"]
fraction_population_per_venue = [1.0, 0.4, 0.4, 1.0]
number_per_venue = [1, 1, 1, 1] #Int.(floor.([1 / 5, 1 / 10] .* n_agents)) #, 1 / 10, 1 / 2] .* n_agents))
graph = generate_random_world_graph(
    n_agents, venues, fraction_population_per_venue, number_per_venue)
true_initial_infected = 0.01
true_gamma = 0.05
true_betas = [0.2, 0.2, 0.2, 0.2]
n_timesteps = 30
#loss = KDELoss(100, BlackBIRDS.MMDKernel())
loss = MSELoss(1e-4)
discrete_sampler = DiffSIR.SM()
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
#loss = GaussianMMDLoss(data, 1e-5)
@model function ppl_model(
        data, true_initial_infected, true_gamma, n_timesteps, graph, discrete_sampler, loss)
    #betas ~ filldist(LogNormal(-1, 0.25), 4)
    #betas ~ filldist(LogNormal(-1, 0.25), 4)
    betas ~ filldist(Uniform(0, 1), 4)
    #log_betas ~ MvNormal(-0.75 * ones(4), 1.0)
    #betas = 10 .^ log_betas
    data ~ SIRJuneModel(
        graph, true_initial_infected, betas, true_gamma, n_timesteps, discrete_sampler, loss)
end
d = 4
#q = make_masked_affine_autoregressive_flow_torch(d, 4, 16);
μ  = -0.4 * ones(d) #randn(d);
L  = LowerTriangular(Diagonal(0.25 .* ones(d)));
q = AdvancedVI.FullRankGaussian(μ, L)
#q = make_planar_flow(d, 5)
#q = make_affine_flow(d, 4, 16);
optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(5e-3))
#optimizer = Optimisers.Adam(1e-3)
prob_model = ppl_model(
    data, true_initial_infected, true_gamma, n_timesteps, graph, discrete_sampler, loss);
q_samples_untrained = rand(q, 10000);

q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 250,
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
    figureSize = 7, truths = [true_betas...],
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
