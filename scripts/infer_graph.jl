using AdvancedVI
using BlackBIRDS
using DiffABM
using Distributions
using DistributionsAD
using DynamicPPL
using Enzyme
using Flux
using Functors
using GraphNeuralNetworks
using Graphs
using LinearAlgebra
using Optimisers
using PyPlot
using StochasticAD
using Zygote

##
struct GraphGenerator{T}
    n_agents::Int64
    venues::Vector{Symbol}
    number_per_venue::Vector{Int64}
    prob_per_venue::Vector{T}
end
#Base.length(generator::GraphGenerator) = length(generator.prob_per_venue)
@functor GraphGenerator (prob_per_venue,)
#Flux.@layer GraphGenerator trainable=(prob_per_venue,)

function to_onehot(index, n)
    diag = Diagonal(ones(n))
    return diag[:, index]
end

function sample_categorical_onehot(probs)
    index = rand(Categorical(probs))
    return to_onehot(index, length(probs))
end

function generate_random_world_graph(
        n_agents, venues, number_per_venue, prob_per_venue)
    tt = typeof(rand(Bernoulli(prob_per_venue[1])))
    n_venues = length(venues)
    num_nodes = Dict(:agent => n_agents)
    eindex_dict = Dict()
    edata = Dict()
    for i in 1:n_venues
        # make complete graph first
        n_this_venue = number_per_venue[i]
        senders = Int64[]
        receivers = Int64[]
        weights = tt[]
        prob_attendance_venue = prob_per_venue[i]
        for j in 1:n_agents
            all_probs = [prob_attendance_venue, ((1.0 - prob_attendance_venue) .*
                                                ones(n_this_venue) ./ n_this_venue)...]
            which_venue = sample_categorical_onehot(all_probs)
            for k in 1:n_this_venue
                push!(senders, j)
                push!(receivers, k)
                push!(weights, which_venue[k+1])
            end
        end
        venue_symbol = venues[i]
        prob_attendance_venue = prob_per_venue[i]
        eindex_dict[(:agent, :attends, venue_symbol)] = (senders, receivers)
        edata[(:agent, :attends, venue_symbol)] = weights
        eindex_dict[(venue_symbol, :attends, :agent)] = (receivers, senders)
        num_nodes[venue_symbol] = n_this_venue
    end
    eindex = (k => v for (k, v) in eindex_dict)
    edata = (k => v for (k, v) in edata)
    return GNNHeteroGraph(eindex; num_nodes, edata)
end

function (generator::GraphGenerator)()
    graph = generate_random_world_graph(
        generator.n_agents, generator.venues, generator.number_per_venue, generator.prob_per_venue)
    return graph
end

function make_sir_abm(
        initial_infected, betas, gamma, graph_generator; backend, discrete_sampler, loss)
    n_timesteps = 30
    delta_t = 1.0
    infection_type = DiffABM.ConstantInfection()
    policies = DiffABM.Policies()
    params = DiffABM.SIRParams(
        graph_generator, [initial_infected], betas, venues, [gamma], delta_t, n_timesteps,
        discrete_sampler, infection_type, policies)

    abm = ABM(params, backend, loss)
    return abm
end
##
n_agents = 500
venues = Symbol.(["household"])#, "company", "leisure"])
number_per_venue = [1]#, 20, 10]
true_initial_infected = 0.1
true_gamma = 0.05
true_betas = [0.8] #, 0.3, 0.1]
discrete_sampler = DiffABM.SM()
true_prob_per_venue = [1.0]#, 0.4, 0.8]
#loss = MSELoss(1)
loss = GaussianMMDLoss(3e-3)
#backend = AutoForwardDiff() #AutoStochasticAD(10)
graph_generator = GraphGenerator(n_agents, venues, number_per_venue, true_prob_per_venue)
backend = AutoStochasticAD(10)
#backend = AutoForwardDiff()
abm = make_sir_abm(true_initial_infected, true_betas, true_gamma, graph_generator;
    backend = backend, discrete_sampler = discrete_sampler, loss = loss)
#@functor GraphGenerator (prob_per_venue,)
#Flux.@layer DiffABM.SIRParams trainable = (initial_infected, venue_betas, gamma, graph_generator)

##

data = rand(abm)

fig, ax = subplots()
ax.plot(data)
fig

##
#using Profile, PProf
#Profile.clear()
v, f = Zygote.pullback(rand, abm);
f(v)
#pprof()

##
v, f = Zygote.pullback(logpdf, abm, data)
f(v)

##
@model function ppl_model(data)
    p ~ filldist(Uniform(0, 1), 6)
    data ~ abm(p)
end
d = 6
μ = -0.4 * ones(d) #randn(d);
L = LowerTriangular(Diagonal(0.25 .* ones(d)));
q = AdvancedVI.FullRankGaussian(μ, L)
#q = AdvancedVI.MeanFieldGaussian(μ, Diagonal(0.25 .* ones(d)))
#q = make_masked_affine_autoregressive_flow_torch(d, 8, 32);
#q = make_planar_flow(d, 5)
#q = make_affine_flow(d, 4, 16);
optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(5e-3))
#optimizer = Optimisers.Adam(1e-3)
prob_model = ppl_model(data);
#q_samples_untrained = rand(q, 10000);

q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 5,
    max_iter = 500,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy()
);
##

##
elbo_vals = [s.elbo for s in stats];
fig, ax = subplots()
ax.plot(elbo_vals)
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples = rand(q, 10000)
prior_samples = rand(Uniform(0, 1), (10, 10000))
#prior = MvNormal(-0.75 * ones(4), 1.0)
#prior_samples = rand(prior, 10000)
pygtc.plotGTC([q_samples', prior_samples'],
    figureSize = 7, truths = [
        true_initial_infected, true_betas..., true_gamma, true_prob_per_venue...],
    chainLabels = ["flow", "prior"])

##
# compare predictions
param_samples = rand(q, 15)
preds = []
for sample in eachcol(param_samples)
    out = rand(abm(sample))
    push!(preds, out)
end

##
fig, ax = subplots()
for pred in preds
    ax.plot(pred, color = "C0", alpha=0.5)
end
ax.plot(data, color = "black", linestyle = "dashed")
fig