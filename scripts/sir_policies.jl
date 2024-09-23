using AdvancedVI
using BlackBIRDS
using Cthulhu
using DiffABM
using Distributions
using DistributionsAD
using DynamicPPL
using Enzyme
using Flux
using ForwardDiff
using Functors
using GraphNeuralNetworks
using Graphs
using LinearAlgebra
using Optimisers
using PyPlot
using StochasticAD
using Zygote

##
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


##
struct ConstantGraph{T}
	graph::T
end
(cg::ConstantGraph)() = cg.graph
@functor SIRParams (policies,)
function make_sir()
	n_agents = 10000
	venues = [:household]#, :school, :leisure]
	number_per_venue = [1]#, 5, 25]
	prob_per_venue = [1.0]#, 0.5, 0.6]
	graph = generate_random_world_graph(n_agents, venues, number_per_venue, prob_per_venue)
	generator = ConstantGraph(graph)
	quarantine = Quarantine([10.0], [20.0], [0.35])
	quarantine_policies = QuarantinePolicies([quarantine])
	policies = Policies(SocialDistancingPolicies(), quarantine_policies)
	initial_infected = [0.05]
	betas = [0.4]#, 0.3, 0.2]
	gamma = [0.05]
	delta_t = 1.0
	n_timesteps = 30
	sir_params = SIRParams(generator, initial_infected, betas, venues, gamma, delta_t, n_timesteps, SAD(), ConstantInfection(), policies)
	return sir_params
end

##
sir_params = make_sir()
sir_abm = ABM(sir_params, AutoStochasticAD(10), GaussianMMDLoss(1e-2))

##
using Profile, PProf

function run(sir_abm)
    v, f = Flux.destructure(sir_abm)
    params = StochasticAD.stochastic_triple.(v)
    return rand(f(params))
end
run(sir_abm);

#Profile.clear()
#@profile run(sir_abm)
#pprof()
@time run(sir_abm)

#@time data = rand(sir_abm)

###
#fig, ax = subplots()
#ax.plot(data)
#fig

###
#@model function ppl_model(data)
	#start_date ~ Uniform(0, 20)
	#end_date ~ Uniform(0, 30)
	#quarantine_prob ~ Uniform(0, 1)
	#data ~ sir_abm([start_date, end_date, quarantine_prob])
#end

#d = 3
#μ = -0.4 * ones(d) #randn(d);
#L = LowerTriangular(Diagonal(0.25 .* ones(d)));
#q = AdvancedVI.FullRankGaussian(μ, L)
#prob_model = ppl_model(data);

#optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(5e-3))

###
##using Profile, PProf
#q, stats = run_vi(
	#model = prob_model,
	#q = q,
	#optimizer = optimizer,
	#n_montecarlo = 5,
	#max_iter = 5,
	#gradient_method = "pathwise",
	#adtype = AutoZygote(),
	#entropy_estimation = AdvancedVI.MonteCarloEntropy(),
#)
###
#pprof()

###
#elbo_vals = [s.elbo for s in stats];
#fig, ax = subplots()
#ax.plot(elbo_vals)
#fig

###
#using PyCall
#pygtc = pyimport("pygtc")
#q_samples = rand(q, 10000);
#start_date_prior_samples = rand(Uniform(0, 20), 10000);
#end_date_prior_samples = rand(Uniform(0, 30), 10000);
#quarantine_prob_prior_samples = rand(Uniform(0, 1), 10000);
#prior_samples = hcat([start_date_prior_samples, end_date_prior_samples, quarantine_prob_prior_samples]...)
#pygtc.plotGTC([q_samples', prior_samples], truths = [10.0, 20.0, 0.25], figureSize = 7)#, chainLabels = ["flow", "prior"])