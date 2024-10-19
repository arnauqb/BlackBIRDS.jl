using AdvancedVI
using BlackBIRDS
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
function make_abm_params(thetas_bounds, initial_efforts_bounds, a, b)
    n_agents = 10000
    n_timesteps = 10
    neighbors = [sample(1:n_agents, 2, replace = false) for _ in 1:n_agents]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(
        n_agents, thetas_bounds, initial_efforts_bounds, neighbors)
    gradient_horizon = 10000
    return DiffABM.AxtellFirmsParams(
        agent_initializer, [a], [b], 0.25, 1.0, n_timesteps, gradient_horizon)
end
params_to_try = [[0.3, 0.7, 0.3, 0.3, 1.5, 2.0], [0.1, 0.8, 0.4, 0.5, 1.5, 3.0]]
true_params = [[0.3, 0.7], [0.1, 0.3], 0.5, 2.0]
true_params_flat = vcat(true_params...)
abm = ABM(make_abm_params(true_params...), AutoForwardDiff(), GaussianMMDLoss(1.0, 5))
data = sum([rand(abm(true_params_flat)) for _ in 1:1]) / 1
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for param in params_to_try
    toplot = mean([rand(abm(param)) for _ in 1:1])
    ax[1].plot(toplot[1, :])
    ax[2].plot(toplot[2, :])
    ax[3].plot(toplot[3, :])
end
ax[1].plot(data[1, :], color = "black", label = "data")
ax[1].set_ylabel("Mean Agent Effort")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[2].set_ylabel("Mean Firm Size")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[3].set_ylabel("Mean Firm Output")
ax[1].legend()
fig

##
v, f = Zygote.pullback(logpdf, abm(true_params_flat), data)
f(v)

##

# test mmd loss point calibration
function evaluate(params, n)
    return -mean(fetch.([Threads.@spawn logpdf(abm([params...]), data) for i in 1:n]))
    #return -mean([logpdf(abm([params...]), data) for i in 1:n])
end
params_train = [0.2, 0.5, 0.2, 0.5, 1.0, 3.0]
n_epochs = 1000
lr = 1e-2
n_samples = 10
rule = Optimisers.Adam(lr)
state_tree = Optimisers.setup(rule, params_train);  # initialise this optimiser's momentum etc.
losses = []
params_history = []
best_params = copy(params_train)
best_loss = Inf
for i in 1:n_epochs
    loss, grads = DifferentiationInterface.value_and_gradient(
        x -> evaluate(x, n_samples), AutoForwardDiff(), params_train)
    state_tree, params_train = Optimisers.update(state_tree, params_train, grads)
    println("Epoch: $i, loss: $(round(loss, digits=3)), params $(round.(params_train, digits=3)), grads $(round.(grads, digits=3))")
    push!(losses, loss)
    push!(params_history, copy(params_train))
    if loss < best_loss
        best_loss = loss
        best_params = copy(params_train)
    end
end

##
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[1].plot(losses)
ax[2].plot(hcat(params_history...)[1, :], label = "a")
ax[2].plot(hcat(params_history...)[2, :], label = "b")
ax[2].plot(hcat(params_history...)[3, :], label = "theta_low")
ax[2].plot(hcat(params_history...)[4, :], label = "theta_high")
ax[2].plot(hcat(params_history...)[5, :], label = "initial_effort_low")
ax[2].plot(hcat(params_history...)[6, :], label = "initial_effort_high")
ax[2].axhline(true_params_flat[1], color = "C0")
ax[2].axhline(true_params_flat[2], color = "C1")
ax[2].axhline(true_params_flat[3], color = "C2")
ax[2].axhline(true_params_flat[4], color = "C3")
ax[2].axhline(true_params_flat[5], color = "C4")
ax[2].axhline(true_params_flat[6], color = "C5")
ax[2].legend()
#ax[1].set_yscale("log")
fig

##
# predictions
n_samples = 10
trained_params = params_history[end]
prior_params = params_history[1]
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for i in 1:n_samples
    trained_pred = rand(abm(best_params))
    prior_pred = rand(abm(prior_params))
    true_pred = rand(abm(true_params_flat))
    ax[1].plot(trained_pred[1, :], color = "C0", alpha = 0.5)
    ax[2].plot(trained_pred[2, :], color = "C0", alpha = 0.5)
    ax[3].plot(trained_pred[3, :], color = "C0", alpha = 0.5)
    ax[1].plot(prior_pred[1, :], color = "C3", alpha = 0.5)
    ax[2].plot(prior_pred[2, :], color = "C3", alpha = 0.5)
    ax[3].plot(prior_pred[3, :], color = "C3", alpha = 0.5)
    ax[1].plot(true_pred[1, :], color = "C1", alpha = 0.5)
    ax[2].plot(true_pred[2, :], color = "C1", alpha = 0.5)
    ax[3].plot(true_pred[3, :], color = "C1", alpha = 0.5)
end
ax[1].set_title("Mean Agent Effort")
ax[2].set_title("Mean Firm Output")
ax[3].set_title("Mean Firm Size")
ax[1].plot(data[1, :], color = "black", label = "data")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[1].plot([], [], color = "C0", alpha = 0.5, label = "trained")
ax[1].plot([], [], color = "C1", alpha = 0.5, label = "true")
ax[1].plot([], [], color = "C3", alpha = 0.5, label = "prior")
ax[1].legend()
fig

##
function transform_params(params)
    return [params[1], params[1] + (1 - params[1]) * params[2], params[3],
        params[3] + (1 - params[3]) * params[4], params[5], params[6]]
end
@model function make_ppl_model(data)
    theta_lower ~ Uniform(0.0, 1.0)
    theta_width ~ Uniform(0.0, 1.0)
    initial_effort_lower ~ Uniform(0.0, 1.0)
    initial_effort_width ~ Uniform(0.0, 1.0)
    a ~ Uniform(0.0, 5.0)
    b ~ Uniform(0.0, 5.0)
    params = transform_params([
        theta_lower, theta_width, initial_effort_lower, initial_effort_width, a, b])
    data ~ abm(params)
end

function run_vi_with_gradient_method(gradient_method)
    Random.seed!(1)
    prob_model = make_ppl_model(data)
    d = 6
    q = make_masked_affine_autoregressive_flow_torch(d, 4, 64)
    #q = create_affine_coupling_flow(d; nlayers=2)
    #q = make_real_nvp_flow_torch(d, 16, 64)
    #q = create_neural_spline_flow(d; n_layers = 2, hdims=64)
    optimizer = Optimisers.AdamW(1e-4, (0.9, 0.99), 1e-5)
    #optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), optimizer)
    q, stats, q_untrained, best_q_cb = run_vi(
        model = prob_model,
        q = q,
        optimizer = optimizer,
        n_montecarlo = 10,
        max_iter = 100,
        gradient_method = gradient_method,
        adtype = AutoZygote(),
        entropy_estimation = AdvancedVI.MonteCarloEntropy()
    )

    best_q = best_q_cb.best_model
    best_elbo = best_q_cb.best_elbo

    return q, stats, q_untrained, best_q_cb, best_q, best_elbo
end
q_score, stats_score, q_untrained_score, best_q_cb_score, best_q_score, best_elbo_score = run_vi_with_gradient_method("score");
q_pathwise, stats_pathwise, q_untrained_pathwise, best_q_cb_pathwise, best_q_pathwise, best_elbo_pathwise = run_vi_with_gradient_method("pathwise");
##
println("best_elbo_score $(best_elbo_score)")
println("best_elbo_pathwise $(best_elbo_pathwise)")
elbo_pathwise = [s.elbo for s in stats_pathwise]
elbo_score = [s.elbo for s in stats_score]
fig, ax = plt.subplots()
ax.plot(-elbo_pathwise, label = "pathwise")
ax.plot(-elbo_score, label = "score")
ax.set_yscale("log")
ax.legend()
best_loss = mean([logpdf(abm(true_params_flat), data) for i in 1:20])
ax.axhline(-best_loss, color = "red")
#ax.set_yscale("log")
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples_score = reduce(hcat, map(transform_params, eachcol(rand(best_q_score, 10000))));
q_samples_pathwise = reduce(hcat, map(transform_params, eachcol(rand(best_q_pathwise, 10000))));
q_samples_untrained = reduce(hcat, map(transform_params, eachcol(rand(q_untrained_score, 10000))));
prior = Product([Uniform(0.0, 1.0), Uniform(0.0, 1.0), Uniform(0.0, 1.0),
    Uniform(0.0, 1.0), Uniform(0.0, 5.0), Uniform(0.0, 5.0)])
prior_samples = reduce(hcat, map(transform_params, eachcol(rand(prior, 10000))));
#prior_samples = transform_params.(prior_samples)
# transform params in each column
fig = pygtc.plotGTC([q_samples_pathwise', q_samples_score', q_samples_untrained'],
    figureSize = 8, truths = vcat(true_params...))
#, truths = true_params,
#paramNames = ["theta_lower", "theta_upper", "initial_effort_lower", "initial_effort_upper", "a", "b"],
#chainLabels = ["trained flow", "untrained flow", "prior"])
#fig.savefig("figures/axtell_posteriors.pdf")

##
# predictive
n_samples = 15
q_samples_score = rand(best_q_score, n_samples);
q_samples_pathwise = rand(best_q_pathwise, n_samples);
q_untrained_samples = rand(q_untrained_score, n_samples);
prior_samples = rand(prior, n_samples)
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
alpha = 0.25
for i in 1:n_samples
    #q_pred_pw = rand(abm(q_samples_pathwise[:, i]))
    q_pred_pw = rand(abm(transform_params(q_samples_pathwise[:, i])))
    ax[1].plot(q_pred_pw[1, :], color = "C0", alpha = 0.5)
    ax[2].plot(q_pred_pw[2, :], color = "C0", alpha = 0.5)
    ax[3].plot(q_pred_pw[3, :], color = "C0", alpha = 0.5)
    q_pred_score = rand(abm(transform_params(q_samples_score[:, i])))
    ax[1].plot(q_pred_score[1, :], color = "C1", alpha = 0.5)
    ax[2].plot(q_pred_score[2, :], color = "C1", alpha = 0.5)
    ax[3].plot(q_pred_score[3, :], color = "C1", alpha = 0.5)

    #true_pred = rand(abm(true_params_flat))
    #ax[1].plot(true_pred[1, :], color = "C1", alpha = alpha)
    #ax[2].plot(true_pred[2, :], color = "C1", alpha = alpha)
    #ax[3].plot(true_pred[3, :], color = "C1", alpha = alpha)
    prior_pred = rand(abm(transform_params(prior_samples[:, i])))
    ax[1].plot(prior_pred[1, :], color = "C3", alpha = alpha)
    ax[2].plot(prior_pred[2, :], color = "C3", alpha = alpha)
    ax[3].plot(prior_pred[3, :], color = "C3", alpha = alpha)
    untrained_pred = rand(abm(transform_params(q_untrained_samples[:, i])))
    ax[1].plot(untrained_pred[1, :], color = "C2", alpha = alpha)
    ax[2].plot(untrained_pred[2, :], color = "C2", alpha = alpha)
    ax[3].plot(untrained_pred[3, :], color = "C2", alpha = alpha)
end
ax[1].plot(data[1, :], color = "black", label = "data")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[3].plot([], [], color = "C0", alpha = 0.5, label = "pathwise")
ax[3].plot([], [], color = "C1", alpha = 0.5, label = "score")
ax[3].plot([], [], color = "C2", alpha = 0.5, label = "untrained")
ax[3].plot([], [], color = "C1", alpha = 0.5, label = "true parameters")
ax[3].plot([], [], color = "C3", alpha = 0.5, label = "prior")
for i in 1:3
    ax[i].set_xlabel("Timestep")
    ax[i].set_yscale("log")
end
ax[3].legend(loc = "center left", bbox_to_anchor = (1, 0.5))
ax[1].set_title("Mean Agent Effort")
ax[2].set_title("Mean Firm Output")
ax[3].set_title("Mean Firm Size")
fig

## HMC
@model function make_unconstrained_model(data)
    a_bijector = inverse(bijector(Uniform(0.0, 5.0)))
    b_bijector = inverse(bijector(Uniform(0.0, 5.0)))
    a_unconstrained ~ Uniform(0.0, 5.0)
    b_unconstrained ~ Uniform(0.0, 5.0)
    #b_unconstrained ~ Normal(0.0, 1.0)
    a = a_bijector(a_unconstrained)
    b = b_bijector(b_unconstrained)
    #a ~ Uniform(0.0, 5.0)
    #b ~ Uniform(0.0, 5.0)
    println("a $(DiffABM.ignore_gradient(a)) b $(DiffABM.ignore_gradient(b))")
    data ~ abm([a, b])
end
##
prob_model_unconstrained = make_unconstrained_model(data)
samples, stats = run_hmc(D = 2, model = prob_model_unconstrained,
    n_samples = 10, n_adapts = 10, initial_p = [0.2, 0.5]);

##
samples_p = reduce(hcat, samples)
#samples_p[1, :] = inverse(bijector(Uniform(0.0, 5.0)))(samples_p[1, :])
#samples_p[2, :] = inverse(bijector(Uniform(0.0, 5.0)))(samples_p[2, :])
prior = Product([Uniform(0.0, 5.0), Uniform(0.0, 5.0)])
prior_samples = rand(prior, 10000)
fig = pygtc.plotGTC([samples_p', prior_samples'], figureSize = 8, paramNames = ["a", "b"],
    truths = true_params, chainLabels = ["HMC", "prior"])

##
bijector_transf = inverse(bijector(prob_model))
#q_transformed = transformed(samples, bijector_transf)
#fig = pygtc.plotGTC(q_transformed', figureSize = 8, paramNames = ["a", "b"], truths = true_params, chainLabels = ["HMC"])

unconstrained_model = prob_model ∘ bijector_transf
lp = DynamicPPL.LogDensityFunction(prob_model)

##