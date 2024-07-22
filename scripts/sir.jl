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
n_agents = 1000
venues = ["household", "company", "school", "leisure"]
fraction_population_per_venue = [1.0, 0.4, 0.4, 1.0]
number_per_venue = Int.(floor.([1/5, 1/10, 1/10, 1/2] .* n_agents))
graph = generate_random_world_graph(
    n_agents, venues, fraction_population_per_venue, number_per_venue)
initial_infected = 0.005
gamma = 0.05
betas = [0.15, 0.2, 0.3, 0.1]
n_timesteps = 30
#loss = KDELoss(20, BlackBIRDS.MMDKernel())
loss = MSELoss(1e-6)
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
#loss = KDELoss(10) #
loss = GaussianMMDLoss(data, 1e-3)
@model function ppl_model(
        data, initial_infected, n_timesteps, graph, discrete_sampler, loss)
    #log_p ~ MvNormal([-1.0, -1.0, -1.0, -1.0, -1.0], 0.5)
    #p = 10 .^ log_p
    #initial_infected = p[1]
    #p = clamp.(p, 1e-6, 2.0)
    p = zeros(5)
    for i in 1:5
        p[i] ~ Beta(1.0, 1.0)
    end
    betas = p[1:end-1]
    gamma = p[end]
    data ~ SIRJuneModel(
        graph, initial_infected, betas, gamma, n_timesteps, discrete_sampler, loss)
end
d = 5
lower_bound = -3.0 * ones(d)
#upper_bound = vcat(0.0, (1.0 .* ones(d - 1))...)
upper_bound = ones(d) #vcat(0.0, (1.0 .* ones(d - 1))...)
q = make_masked_affine_autoregressive_flow_torch(d, 8, 32, param_ranges=[lower_bound, upper_bound]);
q_samples_untrained = rand(q, 10000);
#optimizer = Optimisers.OptimiserChain(Optimisers.AdamW(1e-3), Optimisers.ClipNorm(1.0))
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
prior_samples = rand(MvNormal([-1.0, -1.0, -1.0, -1.0, -1.0], 1.0), 10000);
pygtc.plotGTC([q_samples', prior_samples', q_samples_untrained'],
    figureSize = 7, truths = [log10.(betas)..., log10(gamma)],
    chainLabels=["flow", "prior", "untrained"])
