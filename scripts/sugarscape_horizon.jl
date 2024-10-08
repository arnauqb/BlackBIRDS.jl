using AdvancedVI
using BlackBIRDS
using Bijectors
using DiffABM
using DelimitedFiles
using DifferentiationInterface
using DynamicPPL
using FiniteDifferences
using ForwardDiff
using Distributions
using DistributionsAD
using Functors
using LinearAlgebra
using Optimisers
using StochasticAD
using PyPlot
using Flux
using Random
using Zygote

##
struct SugarScapeModel
	params::SugarScapeParams
end
@functor SugarScapeModel (params,)
@functor SugarScapeParams (agent_initializer,)
@functor TwoPeakBoard ()
#@functor ConstantAgentInitializer (vision,)
function DiffABM.abm_run(model::SugarScapeModel)
    function run(params)
	    board_history, x_history, y_history, wealth_history, alive_history, occupied_history = abm_run(params)
	    wealth_sum = [sum(wealth_history[i]) / (model.params.n_agents * model.params.n_timesteps) for i in 1:model.params.n_timesteps]
        alive_sum = [sum(alive_history[i]) / model.params.n_agents for i in 1:model.params.n_timesteps]
        return vcat(wealth_sum, alive_sum)
    end
    return mean(fetch.([Threads.@spawn run(model.params) for i in 1:1]))
end

function make_model(vision_probs, metabolic_rate_probs; gradient_horizon)
	board_length = 50
	n_agents = 100
	n_timesteps = 50
	#max_sugar = 4.0 #params[1]
	#sugar_regeneration_rate = params[2]
	peak_positions = [
		0.2 * board_length,
		0.2 * board_length,
		0.8 * board_length,
		0.8 * board_length,
		#0.2 * board_length,
		#0.8 * board_length,
		#0.8 * board_length,
		#0.2 * board_length]
    ]
    # load txt as array
    board = readdlm("scripts/sugar-map.txt")[:]
	#distance_function = (a, b) -> sqrt(sum((a .- b) .^ 2) + 1e-3) / 5
	#board_initializer = DiffABM.TwoPeakBoard(
	#	board_length, peak_positions, [max_sugar], distance_function)
	#board = [rand() < 0.01 ? max_sugar : 0.0 for i in 1:board_length*board_length]
	board_initializer = GeneratedBoard(board_length, board)
	positions = [rand(1:board_length, 2) for i in 1:n_agents]
	max_age_distribution = Uniform(60.0, 100.0)
	wealth_distribution = DiscreteUniform(6, 25)
	agent_initializer = RandomAgentInitializer(vision_probs, metabolic_rate_probs, max_age_distribution, wealth_distribution, positions)

	#agent_initializer = ConstantAgentInitializer([params[1]], [2.0], [100.0], [5.0], positions)
	#agent_initializer = DiffABM.RandomAgentInitializer(board_length)
	moving_rule = ArgmaxMovingRule(VonNeumannNeighborhood(board_length, 4))
	sugarscape = SugarScapeParams(
		board_initializer, agent_initializer, moving_rule, board_length,
		n_agents, n_timesteps, [1.0], gradient_horizon)
	return sugarscape
end

##
abm = ABM(SugarScapeModel(make_model([0.0, 0.0, 1.0], [1.0, 0.0], gradient_horizon = 1)), 
    AutoForwardDiff(), MSELoss(0.001))
true_params = [0.1, 0.8, 0.1, 0.75, 0.25]
params_to_try = [
	[1.0, 0.0, 0.0, 0.75, 0.25],
	[0.0, 1.0, 0.0, 0.75, 0.25],
	[0.0, 0.0, 1.0, 0.75, 0.25],
	#[0.0, 1.0, 0.0, 0.25, 0.25],
	#[0.0, 1.0, 0.0, 0.75, 0.25],
]
#params_to_try = [[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]]
ts = [rand(abm(params_to_try[i])) for i in 1:length(params_to_try)]
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
data = rand(abm(true_params))
for i in 1:length(params_to_try)
	ax[1].plot(ts[i][1:length(ts[i])÷2], label = "params= $(params_to_try[i])")
	ax[2].plot(ts[i][length(ts[i])÷2+1:end], label = "params= $(params_to_try[i])")
end
ax[1].plot(data[1:length(data)÷2], label = "data ($true_params)", color = "black")
ax[2].plot(data[length(data)÷2+1:end], label = "data ($true_params)", color = "black")
#ax.legend()
ax[1].set_xlabel("Timestep")
fig

##
v, f = Zygote.pullback(logpdf, abm, rand(abm))
f(v)

## pointwise optimzation
function evaluate_params(params, data, n_samples)
	#ps = softmax(params)
    vision_probs = softmax(params[1:3])
    metabolic_rate_probs = softmax(params[4:5])
    println("vision_probs: $(DiffABM.ignore_gradient.(vision_probs))")
    println("metabolic_rate_probs: $(DiffABM.ignore_gradient.(metabolic_rate_probs))")
    all_probs = vcat(vision_probs, metabolic_rate_probs)
	_, f = Flux.destructure(abm)
	m = f(all_probs)
	asd = -mean(fetch.([Threads.@spawn logpdf(m, data) for i in 1:n_samples]))
    return asd
end
n_epochs = 2000
n_samples = 5
params_train = rand(5)
lr = 1e-2
loss_history = []
params_history = []
gradients_history = []
rule = Optimisers.Adam(lr)  # use the Adam optimiser with its default settings
state_tree = Optimisers.setup(rule, params_train);  # initialise this optimiser's momentum etc.
for i in 1:n_epochs
	loss, gradient = DifferentiationInterface.value_and_gradient(
        x -> evaluate_params(x, data, n_samples), AutoForwardDiff(), params_train)
	push!(gradients_history, gradient)
	state_tree, params_train = Optimisers.update(state_tree, params_train, gradient)
	push!(loss_history, loss)
	push!(params_history, params_train)
	println("Epoch : $i, Loss : $loss, Params : $params_train, Gradient : $gradient")
end
##
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
ax[1].plot(loss_history)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Params")
ps_plot = hcat(params_history...)
ps_plot[1:3, :] = softmax(ps_plot[1:3, :], dims=1)
ps_plot[4:5, :] = softmax(ps_plot[4:5, :], dims=1)
for i in axes(ps_plot, 1)
	ax[2].plot(ps_plot[i, :], color = "C$i")
    ax[2].axhline(true_params[i], color = "C$i", alpha = 0.5)
end
ax[3].set_xlabel("Epoch")
ax[3].set_ylabel("Gradient")
ax[1].set_yscale("log")
fig

##
# test predictions
n_samples = 30
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
params_to_test = ps_plot[:, end]
prior = ps_plot[:, 1]
for i in 1:n_samples
    pred = rand(abm([params_to_test[i] for i in 1:length(params_to_test)]))
    prior_pred = rand(abm([prior[i] for i in 1:length(prior)]))
    true_pred = rand(abm([true_params[i] for i in 1:length(true_params)]))
    ax[1].plot(pred[1:length(pred)÷2], color = "C0", alpha = 0.5)
    ax[2].plot(pred[length(pred)÷2+1:end], color = "C0", alpha = 0.5)
    ax[1].plot(true_pred[1:length(true_pred)÷2], color = "C1", alpha = 0.5)
    ax[2].plot(true_pred[length(true_pred)÷2+1:end], color = "C1", alpha = 0.5)
    ax[1].plot(prior_pred[1:length(prior_pred)÷2], color = "C2", alpha = 0.5)
    ax[2].plot(prior_pred[length(prior_pred)÷2+1:end], color = "C2", alpha = 0.5)
end
ax[1].plot(data[1:length(data)÷2], color = "black")
ax[2].plot(data[length(data)÷2+1:end], color = "black")
fig

##
# bar plot comparing ps_plot with true_params side by side
fig, ax = plt.subplots(figsize=(10, 6))
x = range(1, length(true_params))
width = 0.35

ax.bar(x .- width/2, true_params, width, label="true", alpha=0.8)
ax.bar(x .+ width/2, ps_plot[:, end], width, label="estimated", alpha=0.8)

ax.set_xlabel("Parameter")
ax.set_ylabel("Value")
ax.set_title("Comparison of True and Estimated Parameters")
ax.set_xticks(x)
ax.set_xticklabels([string(i) for i in 1:length(true_params)])
ax.legend()

fig



##
@model function make_ppl_model(data)
	#vision_probs ~ Dirichlet([2.0, 10.0, 5.0, 1.0, 1.0])
	vision_probs ~ Dirichlet([1.0, 1.0, 1.0])
	metabolic_rate_probs ~ Dirichlet([1.0, 1.0])
	all_probs = vcat(vision_probs, metabolic_rate_probs)
	data ~ abm(all_probs)
end
q = make_masked_affine_autoregressive_flow_torch(3, 4, 32);
#d = 3
#μ = zeros(d)
#L = Diagonal(0.25 .* ones(d));
#q = AdvancedVI.MeanFieldGaussian(μ, L)
prob_model = make_ppl_model(data);
#optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.Adam(1e-3))
optimizer = Optimisers.Adam(5e-4)
q, stats, q_untrained, best_q = run_vi(
	model = prob_model,
	q = q,
	optimizer = optimizer,
	n_montecarlo = 5,
	max_iter = 100,
	gradient_method = "pathwise",
	adtype = AutoZygote(),
	entropy_estimation = AdvancedVI.StickingTheLandingEntropy(),
);

##
elbo = [s.elbo for s in stats]
fig, ax = plt.subplots()
ax.plot(elbo)
# estimate best possible loss
best_loss = mean([logpdf(abm(true_params), data) for i in 1:20])
ax.axhline(best_loss, color = "red")
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples = rand(best_q, 10000);
q_samples_untrained = rand(q_untrained, 10000);
#prior = Product([Uniform(0.0, 4.0), Uniform(0.0, 3.0)])
#prior = Uniform(1.0, 3.0)
#prior = Dirichlet(5, 0.2)
#prior1 = Dirichlet(3, 1.0)
#prior2 = Dirichlet(2, 1.0)
prior1 = Dirichlet([1.0, 1.0, 1.0])
prior2 = Dirichlet([1.0, 1.0])
prior_samples = vcat(rand(prior1, 10000), rand(prior2, 10000))
pygtc.plotGTC([q_samples', q_samples_untrained', prior_samples'], figureSize = 8, truths = true_params)#, paramNames = ["", "sugar_regeneration_rate"])
#pygtc.plotGTC([prior_samples'], figureSize = 8, truths = true_params)#, paramNames = ["", "sugar_regeneration_rate"])

##
# predictive
n_samples = 10
q_samples = rand(best_q, n_samples);
q_untrained_samples = rand(q_untrained, n_samples);
idcs = randperm(size(prior_samples, 2))[1:n_samples]
prior_samples = prior_samples[:, idcs]
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
alpha = 0.25 
for i in 1:n_samples
	q_pred = rand(abm([q_samples[:, i]...]))
	ax[1].plot(q_pred[1:length(q_pred)÷2], color = "C0", alpha = 0.50)
	ax[2].plot(q_pred[length(q_pred)÷2+1:end], color = "C0", alpha = 0.50)
	true_pred = rand(abm([true_params[i] for i in 1:length(true_params)]))
	#ax[1].plot(true_pred[1:length(true_pred)÷2], color = "C1", alpha = alpha)
	#ax[2].plot(true_pred[length(true_pred)÷2+1:end], color = "C1", alpha = alpha)
	prior_pred = rand(abm([prior_samples[:, i]...]))
	ax[1].plot(prior_pred[1:length(prior_pred)÷2], color = "C3", alpha = alpha)
	ax[2].plot(prior_pred[length(prior_pred)÷2+1:end], color = "C3", alpha = alpha)
	ax[1].plot(data[1:length(data)÷2], color = "black")
	ax[2].plot(data[length(data)÷2+1:end], color = "black")
end
for i in 1:2
	ax[i].plot([], [], color = "C0", alpha = 0.5, label = "q")
	ax[i].plot([], [], color = "C1", alpha = 0.5, label = "true")
	ax[i].plot([], [], color = "C3", alpha = 0.5, label = "prior")
	ax[i].legend()
end
fig


##
abm = ABM(SugarScapeModel(make_model([5.0], gradient_horizon = 101)), AutoForwardDiff(), MSELoss(0.01))
data = rand(abm([5.0]))
function sample(params)
	_, rec_f = Flux.destructure(abm)
	return rand(rec_f(params))
end
function compute_loss(params, data)
	return sum((sample(params) - data) .^ 2)
end
function compute_average_loss(params, data, n_samples)
	Random.seed!(0)
	return mean(fetch.([Threads.@spawn compute_loss(params, data) for i in 1:n_samples]))
end
##
params = [2.0]
n_samples = 50
grads = vcat([ForwardDiff.gradient(x -> compute_loss(x, data), params) for i in 1:n_samples]...)
grads_sad = vcat([StochasticAD.derivative_estimate(x -> compute_loss(x, data), params) for i in 1:n_samples]...)
##
#m = FiniteDifferences.central_fdm(5, 1)
#grads_fd = DifferentiationInterface.gradient(x -> compute_average_loss(x, data, 100), AutoFiniteDifferences(m), params)

##


fig, ax = plt.subplots()
ax.boxplot([grads, grads_sad], labels = ["ForwardDiff", "StochasticAD"], showfliers = false)
#ax.boxplot([grads_sad], labels = ["StochasticAD"], showmeans = true, showfliers = false)
#ax.axhline(grads_fd[1], color = "red")
#ax.boxplot(grads_fd, showfliers = false)
fig
##

using PyCall
animation = pyimport("matplotlib.animation")
function run_and_animate(params)
	function update_plot(frame)
		frame = frame + 1
		ax.clear()
		board = board_history[frame]
		x = x_history[frame]
		y = y_history[frame]
		alive = alive_history[frame]

		# Use pcolormesh with plasma colormap
		im = ax.pcolormesh(board', cmap = "inferno", vmin = 0, vmax = 10.0)
		#fig.colorbar(im, ax=ax)

		# Scatter plot with white color for alive and black for dead
		alive_color = [a == 1.0 ? "white" : "black" for a in alive]
		ax.scatter(x, y, c = alive_color, s = 10, edgecolors = "none")

		ax.set_title("Timestep $frame")
		ax.set_xlim(0, size(board, 1))
		ax.set_ylim(0, size(board, 2))
		return [im]
	end

	Random.seed!(1234)
    vision_probs = params[1:3]
    metabolic_rate_probs = params[4:5]
	model = make_model(vision_probs, metabolic_rate_probs, gradient_horizon = 101)
	board_history, x_history, y_history, wealth_history, alive_history, occupied_history = abm_run(model)

	max_sugar = maximum(model.board_initializer.board)

	# Create the figure and axis outside the animation function
	fig, ax = plt.subplots()
	board = board_history[1]
	im = ax.pcolormesh(board', cmap = "inferno", vmin = 0, vmax = 5.0)
	fig.colorbar(im, ax = ax)
	# Create the animation
	anim = animation.FuncAnimation(
		fig,
		update_plot,
		frames = model.n_timesteps,
		interval = 100,  # 100 ms between frames
		blit = true,
		repeat = false,
	)

	# Save the animation
	anim.save("sugarscape.gif", writer = "pillow", fps = 10)
	plt.close(fig)
end
run_and_animate(true_params)

