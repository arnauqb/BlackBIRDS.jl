using AdvancedVI
using Distributions
using DiffSIR
using DynamicPPL
using Functors
using Flux
using LinearAlgebra
using CairoMakie
using Optimisers
using Zygote

using BlackBIRDS
using BlackBIRDS.SIRJune

##
n_agents = 500
venues = ["household"] #, "company", "school", "leisure"]
fraction_population_per_venue = [1.0] #, 0.4, 0.4, 1.0]
number_per_venue = [1] #Int.(floor.([1/5, 1/10, 1/10, 1/2] .* n_agents))
graph = generate_random_world_graph(n_agents, venues, fraction_population_per_venue, number_per_venue)
initial_infected = 0.02
gamma = 0.05
betas = [0.3] #, 0.2, 0.2, 0.1]
n_timesteps = 60
loss = MSELoss( 1e-6)
discrete_sampler = DiffSIR.ST()
sir = SIRJuneModel(graph, initial_infected, betas, gamma, n_timesteps, discrete_sampler, loss);
@functor SIRJuneModel (venue_betas, gamma)

##

data = rand(sir)

fig = Figure()
fig, lines(fig[1,1], data[1, :])
lines!(fig[1,1], data[2, :])
fig

##

@model function ppl_model(data, initial_infected, n_timesteps, graph, discrete_sampler, loss)
    #log_p ~ MvNormal([-2.5, -1.0, -1.0, -1.0, -1.0, -1.0], 1.0)
    log_p ~ MvNormal([-1.0, -1.0], 0.5)
    p = 10 .^ log_p
    p = clamp.(p, 0.0, 1.0)
    data ~ SIRJuneModel(graph, initial_infected, [p[1]], p[2], n_timesteps, discrete_sampler, loss)
end
d = 2
mu = 1.0 .* ones(d)
L = Diagonal(ones(d)) |> LowerTriangular
q = FullRankGaussian(mu, L)
#q = make_masked_affine_autoregressive_flow_torch(d, 4, 32);
#q = make_planar_flow(d, 10)
q_samples_untrained = rand(q, 10000);
optimizer = Optimisers.OptimiserChain(Optimisers.AdamW(1e-3), Optimisers.ClipNorm(1.0))
prob_model = ppl_model(data, initial_infected, n_timesteps, graph, discrete_sampler, loss);

##
q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 1000,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy(),
);

##
elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

##
using PairPlots

q_samples = rand(q, 10000)
prior_samples = rand(MvNormal([-2.5, -1.0, -1.0, -1.0, -1.0, -1.0], 1.0), 10000);
function make_table(samples)
    return (;
        #i0 = samples[1, :],
        gamma = samples[2, :],
        beta1 = samples[1, :],
        #beta2 = samples[4, :],
        #beta3 = samples[5, :],
        #beta4 = samples[6, :],
    )
end
table = make_table(q_samples);
table_prior = make_table(prior_samples);
table_untrained = make_table(q_samples_untrained);
#truths = (; i0 = log10(p[1]), gamma = log10(p[2]), beta1 = log10(p[3]), beta2 = log10(p[4]), beta3 = log10(p[5]), beta4 = log10(p[6]))
truths = (; beta1=betas[1], gamma=gamma)
c1 = Makie.wong_colors(0.5)[1];
c2 = Makie.wong_colors(0.5)[2];
c3 = Makie.wong_colors(0.5)[3];
pairplot(
    PairPlots.Series(table, label="Trained", color=c1, strokecolor=c1),
    PairPlots.Series(table_prior, label="Prior", color=c2, strokecolor=c2),
    PairPlots.Series(table_untrained, label="Untrained", color=c3, strokecolor=c3),
    PairPlots.Truth(truths, color = "black"),
)
