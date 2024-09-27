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
struct TrainableCategorical{T}
	p::T
	# constructor normalizes probs to 1.
	function TrainableCategorical(p::T) where {T}
		p = p ./ sum(p)
		return new{T}(p)
	end
end
@functor TrainableCategorical (p,)
Base.rand(d::TrainableCategorical) = rand(Categorical(d.p))

@functor SugarScapeModel (params,)
@functor SugarScapeParams (agent_initializer,)
function DiffABM.abm_run(model::SugarScapeModel)
	#Random.seed!(1234)
	_, _, _, alive_history = DiffABM.abm_run(model.params)
	return [sum(alive_history[i]) for i in 1:model.params.n_timesteps]
end
function make_model(max_sugar, regeneration_rate, vision_probs)
	board_length = 100
	n_agents = 100
	n_timesteps = 30
	peak_positions = [[5, 5], [95, 95]]
	distance = (x, y) -> sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2) / 1
	vision_distribution = TrainableCategorical(vision_probs)
	board_initializer = TwoPeakBoard(board_length, peak_positions, [max_sugar], distance)
	agent_initializer = RandomAgentInitializer(board_length, vision_distribution = vision_distribution)
	moving_rule = ArgmaxMovingRule(VonNeumannNeighborhood(board_length, length(vision_probs)))
	sugarscape = SugarScapeParams(board_initializer, agent_initializer, moving_rule, board_length, n_agents, n_timesteps, [regeneration_rate])
	return sugarscape
end
true_sugar_regeneration_rate = 0.5
true_max_sugar = 10.0
true_vision_probs = [0.00, 0.0, 0.0, 0.0, 1.0]
sugarscape = ABM(SugarScapeModel(make_model(true_max_sugar, true_sugar_regeneration_rate, true_vision_probs)), AutoForwardDiff(), MSELoss(0.1))

##
true_alive = rand(sugarscape)
tests_values = [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
fig, ax = plt.subplots()
for test_value in tests_values
	ax.plot(rand(sugarscape(test_value)), label = "vision_probs = $test_value")
end
ax.plot(true_alive, label = "true")
ax.set_ylim(0, 105)
ax.legend()
fig

##
@model ppl_model(alive_history) = begin
	vision_probs ~ Dirichlet(5, 1.0)
	# numerical error may cause the sum of the probabilities to be slightly greater than 1
	vision_probs = vision_probs ./ sum(vision_probs)
	#max_sugar ~ Uniform(1.0, 30.0)
	#regeneration_rate ~ Uniform(0.0, 1.0)
	alive_history ~ sugarscape(vision_probs)
end
d = 4
#μ = zeros(d)
#L = Diagonal(0.25 .* ones(d));
#q = AdvancedVI.MeanFieldGaussian(μ, L)
q = make_masked_affine_autoregressive_flow_torch(d, 8, 32);
prob_model = ppl_model(true_alive);

#optimizer = Optimisers.AdamW(1e-3)
optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(1e-3))

q, stats = run_vi(
	model = prob_model,
	q = q,
	optimizer = optimizer,
	n_montecarlo = 10,
	max_iter = 200,
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
#prior = Product([Uniform(1.0, 30.0), Uniform(0.0, 1.0)])
prior = Dirichlet(5, 1.0)
prior_samples = rand(prior, 10000)
#pygtc.plotGTC([q_samples', prior_samples'], truths = [true_max_sugar, true_sugar_regeneration_rate], figureSize = 4)#, chainLabels = ["flow", "prior"])
pygtc.plotGTC([q_samples', prior_samples'], truths = [true_vision_probs...], figureSize = 4)#, chainLabels = ["flow", "prior"])

##
# test posterior predictive
fig, ax = subplots()
n_samples = 50
post_samples = rand(q, n_samples);
prior_samples = rand(prior, n_samples)
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

##
function category_to_onehot(category, n_categories)
	# do a soft approximation with a softmax

end
