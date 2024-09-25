using AdvancedVI
using BlackBIRDS
using DiffABM
using DynamicPPL
using Distributions
using DistributionsAD
using Functors
using LinearAlgebra
using Optimisers
using PyPlot
using Flux
using Random

##
struct SugarScapeModel
	params::SugarScapeParams
end
@functor SugarScapeModel (params,)
@functor SugarScapeParams (board_initializer, sugar_regeneration_rate)
function DiffABM.abm_run(model::SugarScapeModel)
	#Random.seed!(1234)
	_, _, _, alive_history = DiffABM.abm_run(model.params)
	return [sum(alive_history[i]) for i in 1:model.params.n_timesteps]
end
function make_model(max_sugar, regeneration_rate)
	board_length = 35
	n_agents = 50
	n_timesteps = 50
	peak_positions = [[10, 40], [40, 10]]
	distance = (x, y) -> sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2) / 5
	board_initializer = TwoPeakBoard(board_length, peak_positions, [max_sugar], distance)
	agent_initializer = RandomAgentInitializer(board_length)
	moving_rule = ArgmaxMovingRule(VonNeumannNeighborhood)
	sugarscape = SugarScapeParams(board_initializer, agent_initializer, moving_rule, board_length, n_agents, n_timesteps, [regeneration_rate])
	return sugarscape
end
true_sugar_regeneration_rate = 0.5
true_max_sugar = 150.0
sugarscape = ABM(SugarScapeModel(make_model(true_max_sugar, true_sugar_regeneration_rate)), AutoForwardDiff(), MSELoss(1.0))

##
true_alive = rand(sugarscape)
fig, ax = plt.subplots()
ax.plot(true_alive)
fig

##
@model ppl_model(alive_history) = begin
	max_sugar ~ Uniform(10.0, 1000.0)
	regeneration_rate ~ Uniform(0.0, 2.0)
	alive_history ~ sugarscape([max_sugar,regeneration_rate])
end
d = 2
#μ = zeros(d)
#L = Diagonal(0.25 .* ones(d));
#q = AdvancedVI.MeanFieldGaussian(μ, L)
q = make_masked_affine_autoregressive_flow_torch(d, 4, 32)
prob_model = ppl_model(true_alive);

optimizer = Optimisers.AdamW(1e-3)
q, stats = run_vi(
	model = prob_model,
	q = q,
	optimizer = optimizer,
	n_montecarlo = 10,
	max_iter = 100,
	gradient_method = "pathwise",
	adtype = AutoZygote(),
	entropy_estimation = AdvancedVI.StickingTheLandingEntropy(),
);

##
elbo_vals = [s.elbo for s in stats];
fig, ax = subplots()
ax.plot(elbo_vals)
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples = rand(q, 10000);
#prior_samples = rand(Uniform(0.0, 10.0), (1, 10000))
prior_samples = rand(Product([Uniform(10.0, 1000.0), Uniform(0.0, 2.0)]), 10000)
pygtc.plotGTC([q_samples', prior_samples'], truths = [true_max_sugar, true_sugar_regeneration_rate], figureSize = 4)#, chainLabels = ["flow", "prior"])

##
# test posterior predictive
fig, ax = subplots()
n_samples = 50
post_samples = rand(q, n_samples);
#prior_samples = rand(Uniform(10.0, 1000.0), (1, n_samples))
prior_samples = rand(Product([Uniform(10.0, 1000.0), Uniform(0.0, 2.0)]), n_samples)
preds = []
for i in 1:n_samples
	post_model = sugarscape(post_samples[:, i])
	prior_model = sugarscape(prior_samples[:, i])
	pred = rand(post_model)
	prior_pred = rand(prior_model)
	ax.plot(pred, color = "C0", alpha = 0.25)
	ax.plot(prior_pred, color = "C1", alpha = 0.25)
end
ax.plot([], [], color = "C0", label = "posterior")
ax.plot([], [], color = "C1", label = "prior")
ax.legend()
ax.plot(true_alive, color = "black")
fig
