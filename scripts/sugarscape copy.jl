using AdvancedVI
using BlackBIRDS
using DiffABM
using DynamicPPL
using Distributions
using Flux
using DifferentiationInterface
using DistributionsAD
using Functors
using LinearAlgebra
using ProgressMeter

using Optimisers
using PyPlot
using Flux
using Random

##
struct SugarScapeModel
	params::SugarScapeParams
end
@functor SugarScapeModel (params, )
@functor SugarScapeParams (board_initializer,)
@functor DiffABM.TwoPeakBoard (sugar_peaks, )
function DiffABM.abm_run(model::SugarScapeModel)
	#Random.seed!(1234)
    #return DiffABM.abm_run(model.params)
    _, x_history, y_history, _ = DiffABM.abm_run(model.params)
    #return [x_history[end], y_history[end]]
    #return [mean((x_history[end] .- 5.0) .^ 2)]
    #return [distance_to_line]
    return [mean((x_history[end] .- 5.0) .^ 2 + (y_history[end] .- 5.0) .^ 2)]
end

function make_model(peak_positions)
	board_length = 20
	n_agents = 20
	n_timesteps = 10
    sugar_regeneration_rate = 1.0
    vision_probs = zeros(2) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]
    vision_probs[end] = 1.0
	vision_distribution = Categorical(vision_probs)
    max_sugar = 10.0
    distance_function = (a, b) -> sqrt(sum((a .- b) .^ 2) + 1e-3)
    board_initializer = DiffABM.TwoPeakBoard(board_length, peak_positions, [max_sugar], distance_function)
    #board_initializer = DiffABM.GeneratedBoard(board_length, board)
	agent_initializer = RandomAgentInitializer(board_length, vision_distribution = vision_distribution)
	moving_rule = ArgmaxMovingRule(MooreNeighborhood(board_length, length(vision_probs)))
	sugarscape = SugarScapeParams(board_initializer, agent_initializer, moving_rule, board_length, n_agents, n_timesteps, [sugar_regeneration_rate])
	return sugarscape
end
##
#perfect_board = zeros(10, 10)
#perfect_board[5, :] .= 10.0
#perfect_board = perfect_board[:]
perfect_peaks = [5.0, 5.0]
sugarscape_params = make_model(perfect_peaks)
sugarscape = ABM(SugarScapeModel(sugarscape_params), AutoForwardDiff(), MSELoss(1.0))
board = DiffABM.initialize_board(sugarscape_params.board_initializer, Float64)
rand(sugarscape)

##
fig, ax = plt.subplots()
ax.pcolormesh(1:10, 1:10, board)
fig
##
#_, x, y, _ = y = rand(sugarscape)
x, y = rand(sugarscape)
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
fig


##
test = sugarscape([2.0, 8.0])
v, f = Zygote.pullback(rand, test)
f(v)




##
# training loop
#d = 10 * 10
#q = Dense(1 => d) #make_masked_affine_autoregressive_flow_torch(d, 4, 32);
#vq, f = Flux.destructure(q)
losses_hist = []
n_epochs = 100
n_batch = 1
random_peaks = [2.0, 8.0]
#rule = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(1e-3))
#rule = Optimisers.AdamW(1e-3)
#state_tree = Optimisers.setup(rule, q);
function evaluate_flow(peaks, n_samples)
    ret = 0.0
    for i in 1:n_samples
        #board = q([1.0]) #rand(q)
        #board = 10 .* (board .^ 2)
        model = sugarscape(peaks)
        ret += rand(model)[1]
    end
    return ret / n_samples
end
#best_q = copy(vq)
best_loss = Inf
p = Progress(n_epochs)
peaks = random_peaks
lr = 1e-2
for i in 1:n_epochs
    value, grads = Flux.withgradient(evaluate_flow, peaks, n_batch)
    #println("grads ", grads[1])
    push!(losses_hist, value)
    if value < best_loss
        best_loss = value
        #v, _ = Flux.destructure(q)
        #best_q = copy(vq)
        best_peaks = copy(peaks)
    end
    g = grads[1]
    #g = grads[1] ./ sqrt(sum(grads[1] .^ 2))
    next!(p, showvalues = [(:loss, value)])
    # update optimiser
    #state_tree, q = Optimisers.update(state_tree, q, g)
    println("g ", g)
    println("peaks ", peaks)

    peaks = peaks - lr * g
end
##
fig, ax = subplots()
ax.plot(losses_hist)
fig

##
bq = f(best_q)
fit_board = 10 .* (bq([1.0]) .^ 2)
fit_board_img = reshape(fit_board, 10, 10)
fig, ax = plt.subplots()
ax.pcolormesh(1:10, 1:10, fit_board_img')
fig

##
#board = 100 * q(rand(1)) .^ 2 #rand(q) .^ 2
#fig, ax = subplots()
#board_img = reshape(board, 10, 10)
#ax.pcolormesh(1:10, 1:10, board_img')
#fig
fitted_model = make_model(best_peaks);
unfitted_model = make_model(random_peaks);
_, x_fitted, y_fitted, _ = abm_run(fitted_model)
_, x_unfitted, y_unfitted, _ = abm_run(unfitted_model)
fig, ax = subplots()
ax.scatter(x_fitted[end], y_fitted[end])
ax.scatter(x_unfitted[end], y_unfitted[end], color = "red")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
fig

##
@model ppl_model(data) = begin
    board ~ filldist(LogNormal(0, 1), 10 * 10)
    #board = reshape(board, 100, 100)
	data ~ sugarscape(board)
end
d = 10 * 10
#μ = zeros(d)
#L = Diagonal(0.25 .* ones(d));
#q = AdvancedVI.MeanFieldGaussian(μ, L)
data = [0.0]
q = make_masked_affine_autoregressive_flow_torch(d, 8, 32);
prob_model = ppl_model(data);
#
optimizer = Optimisers.AdamW(1e-3)
#optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(1e-3))

q, stats = run_vi(
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
elbo_vals = [s.elbo for s in stats];
fig, ax = subplots()
ax.plot(elbo_vals)
fig

##
#using PyCall
#pygtc = pyimport("pygtc")
#q_samples = rand(q, 10000);
##prior = Product([Uniform(1.0, 30.0), Uniform(0.0, 1.0)])
#prior = Dirichlet(5, 1.0)
#prior_samples = rand(prior, 10000)
##pygtc.plotGTC([q_samples', prior_samples'], truths = [true_max_sugar, true_sugar_regeneration_rate], figureSize = 4)#, chainLabels = ["flow", "prior"])
#pygtc.plotGTC([q_samples', prior_samples'], truths = [true_vision_probs...], figureSize = 4)#, chainLabels = ["flow", "prior"])

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

board = rand(q)
board_img = reshape(board, 10, 10)
fig, ax = subplots()
ax.pcolormesh(1:10, 1:10, board_img')
fig