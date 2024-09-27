using AdvancedVI
using BlackBIRDS
using DiffABM
using DynamicPPL
using Distances
using Distributions
using Flux
using DifferentiationInterface
using DistributionsAD
using Functors
using LinearAlgebra
using ProgressMeter

using Optimisers
using PyPlot
using Plots
using Flux
using Random

##
struct SugarScapeModel
	params::SugarScapeParams
end
@functor SugarScapeModel (params, )
@functor SugarScapeParams (board_initializer,)
@functor DiffABM.TwoPeakBoard (sugar_peaks, )
function circularity(x_history, y_history)
    centre = [10.0, 10.0]
    distances = pairwise(SqEuclidean(), [centre], [[x,y] for (x,y) in zip(x_history, y_history)])[:]
    distances = sqrt.(distances .+ 1e-3)
    mean_distance = mean(distances)
    variance = mean((distances .- mean_distance) .^ 2)
    circularity = variance / mean_distance
    return circularity
end
function DiffABM.abm_run(model::SugarScapeModel)
	#Random.seed!(1234)
    #return DiffABM.abm_run(model.params)
    _, x_history, y_history, _ = DiffABM.abm_run(model.params)
    #return [x_history[end], y_history[end]]
    return [-circularity(x_history[end], y_history[end])]
    #return [mean((x_history[end] .- 5.0) .^ 2)]
    #return [distance_to_line]
    #return [(mean((x_history[end] .- 5.0) .^ 2 + (y_history[end] .- 5.0) .^ 2) - 1.0) .^ 2]
end

function make_model(peak_positions)
	board_length = 20
	n_agents = 20
	n_timesteps = 20
    sugar_regeneration_rate = 0.1
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
perfect_peaks = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
sugarscape_params = make_model(perfect_peaks)
sugarscape = ABM(SugarScapeModel(sugarscape_params), AutoForwardDiff(), MSELoss(0.01))
board = DiffABM.initialize_board(sugarscape_params.board_initializer, Float64)
rand(sugarscape)

##
v, f = Zygote.pullback(rand, sugarscape)
f(v)

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
random_peaks = [2.0, 8.0, 3.0, 4.0, 6.0, 2.0]
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
best_peaks = copy(peaks)
lr = 1e-4
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
# draw circle of radius 5
circle = plt.Circle((5, 5), 1, color = "red", fill = false)
ax.add_artist(circle)
ax.scatter(x_unfitted[end], y_unfitted[end], color = "red")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
fig

##
@model ppl_model(data) = begin
    peaks ~ filldist(Uniform(1.0, 20.0), 10)
    #board = reshape(board, 100, 100)
	data ~ sugarscape(peaks)
end
d = 10
#μ = zeros(d)
#L = Diagonal(0.25 .* ones(d));
#q = AdvancedVI.MeanFieldGaussian(μ, L)
data = [0.0]
q = make_masked_affine_autoregressive_flow_torch(d, 4, 16);
prob_model = ppl_model(data);
#
optimizer = Optimisers.AdamW(1e-3)
#optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.AdamW(1e-3))

q, stats, q_untrained = run_vi(
	model = prob_model,
	q = q,
	optimizer = optimizer,
	n_montecarlo = 10,
	max_iter = 1000,
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
q_samples_untrained = rand(q_untrained, 10000);
#prior = Product([Uniform(1.0, 30.0), Uniform(0.0, 1.0)])
prior = Uniform(1.0, 20.0) #Dirichlet(5, 1.0)
prior_samples = rand(prior, (10, 10000))
#pygtc.plotGTC([q_samples', prior_samples'], truths = [true_max_sugar, true_sugar_regeneration_rate], figureSize = 4)#, chainLabels = ["flow", "prior"])
pygtc.plotGTC([q_samples', prior_samples', q_samples_untrained'],  figureSize = 4)#, chainLabels = ["flow", "prior"])

##
peaks = rand(q)
fitted_model = make_model(peaks)
fitted_board = DiffABM.initialize_board(fitted_model.board_initializer, Float64)
fig, ax = subplots()
ax.pcolormesh(1:size(fitted_board, 1), 1:size(fitted_board, 2), fitted_board')
fig

##
board_history, x_history, y_history, alive_history = DiffABM.abm_run(fitted_model)
function plot_board(board, x, y, alive)
    p = Plots.plot()
    board_x = collect(1:size(board)[1])
    board_y = collect(1:size(board)[2])
    heatmap!(p, board_x, board_y, board', clim = (0, 10.0))
    # scatter with white color for alive and red for dead
    alive_color = [Bool(alive) ? :white : :black for alive in alive]
    scatter!(p, x, y, c=alive_color, ms=2)
    p
end
plot_board(board_history[1], x_history[1], y_history[1], alive_history[1])
anim = @animate for i in 1:10
    plot_board(board_history[i], x_history[i], y_history[i], alive_history[i])
end
gif(anim, "sugarscape.gif", fps=1)