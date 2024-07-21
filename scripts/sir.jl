using AdvancedVI
using Distributions
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
n_agents = 100
venues = ["household"] #, "company", "school", "leisure"]
fraction_population_per_venue = [1.0] #, 0.4, 0.4, 1.0]
number_per_venue = [1] #Int.(floor.([1/5, 1/10, 1/10, 1/2] .* n_agents))
graph = generate_random_world_graph(
    n_agents, venues, fraction_population_per_venue, number_per_venue)
initial_infected = 0.1
gamma = 0.1
betas = [0.5] #, 0.2, 0.2, 0.1]
n_timesteps = 30
loss = KDELoss(20, BlackBIRDS.MMDKernel())
discrete_sampler = DiffSIR.SM()
sir = SIRJuneModel(
    graph, initial_infected, betas, gamma, n_timesteps, discrete_sampler, loss);
@functor SIRJuneModel (venue_betas, gamma)

##

data = rand(sir)

fig, ax = subplots()
ax.plot(data[1, :], label = "Infected")
ax.plot(data[2, :], label = "Recovered")
ax.legend()
fig

##
#loss = KDELoss(10) #GaussianMMDLoss(data, 0.01)
@model function ppl_model(
        data, initial_infected, n_timesteps, graph, discrete_sampler, loss)
    #log_p ~ MvNormal([-2.5, -1.0, -1.0, -1.0, -1.0, -1.0], 1.0)
    log_p ~ MvNormal([-1.5, -1.5], 1.0)
    p = 10 .^ log_p
    #p = clamp.(p, 0.0, 5.0)
    data ~ SIRJuneModel(
        graph, initial_infected, [p[1]], p[2], n_timesteps, discrete_sampler, loss)
end
d = 2
q = make_masked_affine_autoregressive_flow_torch(d, 4, 16)#, param_ranges=[[-3.0, -3.0], [2.0, 2.0]]);
q_samples_untrained = rand(q, 10000);
#optimizer = Optimisers.OptimiserChain(Optimisers.AdamW(1e-1), Optimisers.ClipNorm(1.0))
optimizer = Optimisers.AdamW(1e-3)
prob_model = ppl_model(data, initial_infected, n_timesteps, graph, discrete_sampler, loss);

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
prior_samples = rand(MvNormal([-1.5, -1.5], 1.0), 10000);
pygtc.plotGTC([q_samples', prior_samples', q_samples_untrained'],
    figureSize = 7, truths = [log10(betas[1]), log10(gamma)],
    chainLabels=["flow", "prior", "untrained"])

##
#q_samples = rand(q, 10000)
#prior_samples = rand(MvNormal([-2.5, -1.0, -1.0, -1.0, -1.0, -1.0], 1.0), 10000);
#function make_table(samples)
#    return (;
#        #i0 = samples[1, :],
#        beta1 = samples[1, :],
#        gamma = samples[2, :],
#        #beta2 = samples[4, :],
#        #beta3 = samples[5, :],
#        #beta4 = samples[6, :],
#    )
#end
#table = make_table(q_samples);
#table_prior = make_table(prior_samples);
#table_untrained = make_table(q_samples_untrained);
##truths = (; i0 = log10(p[1]), gamma = log10(p[2]), beta1 = log10(p[3]), beta2 = log10(p[4]), beta3 = log10(p[5]), beta4 = log10(p[6]))
#truths = (; beta1=log10(betas[1]), gamma=log10(gamma))
#c1 = Makie.wong_colors(0.5)[1];
#c2 = Makie.wong_colors(0.5)[2];
#c3 = Makie.wong_colors(0.5)[3];
#pairplot(
#    PairPlots.Series(table, label="Trained", color=c1, strokecolor=c1),
#    PairPlots.Series(table_prior, label="Prior", color=c2, strokecolor=c2),
#    PairPlots.Series(table_untrained, label="Untrained", color=c3, strokecolor=c3),
#    PairPlots.Truth(truths, color = "black"),
#)
#