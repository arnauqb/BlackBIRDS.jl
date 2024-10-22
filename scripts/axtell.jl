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
using ProgressMeter
using StochasticAD
using PyPlot
using Flux
using Random
using Zygote
using PyCall
pyimport("scienceplots")
plt.style.use("science")

##
@functor DiffABM.RandomAxtellAgentInitializer (thetas_bounds, initial_efforts_bounds)
function make_abm_params(thetas_bounds, initial_efforts_bounds, a, b)
    n_agents = 1000
    n_timesteps = 30
    neighbors = [sample(1:n_agents, 2, replace = false) for _ in 1:n_agents]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(
        n_agents, thetas_bounds, initial_efforts_bounds, neighbors)
    gradient_horizon = 200
    return DiffABM.AxtellFirmsParams(
        agent_initializer, [a], [b], 0.25, 1.0, n_timesteps, gradient_horizon)
end
params_to_try = [[0.3, 0.7, 0.4, 0.5, 0.25, 1.0], [0.1, 0.2, 0.2, 0.8, 0.25, 1.0]]
true_params = [[0.3, 0.7], [0.4, 0.75], 0.25, 1.0]
true_params_flat = vcat(true_params...)
#my_loss = MSELoss(w = 1.0, n_samples = 1)
my_loss = GaussianMMDLoss(w = 0.1, n_samples = 8)
abm = ABM(make_abm_params(true_params...), AutoForwardDiff(), my_loss)
data = sum([rand(abm(true_params_flat)) for _ in 1:10]) / 10
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for param in params_to_try
    toplot = mean([rand(abm(param)) for _ in 1:1])
    ax[1].plot(toplot[1, :])
    #ax[2].plot(toplot[2, :])
    #ax[3].plot(toplot[3, :])
end
ax[1].plot(data[1, :], color = "black", label = "data")
ax[1].set_ylabel("Mean Firm Output")
#ax[2].plot(data[2, :], color = "black", label = "data")
#ax[2].set_ylabel("Mean Firm Size")
#ax[3].plot(data[3, :], color = "black", label = "data")
#ax[2].set_ylabel("Mean Agent Effort")
ax[1].legend()
fig

##
Random.seed!(0)
x = rand(6)
v, f = Zygote.pullback(logpdf, abm(x), data)
f(v)

##

# test mmd loss point calibration
function evaluate(params, n)
    #return -mean(fetch.([Threads.@spawn logpdf(abm(params), data) for i in 1:n]))
    return -mean([logpdf(abm([params...]), data) for i in 1:n])
end
params_train = [0.2, 0.5, 0.2, 0.8, 0.5, 0.5]
n_epochs = 1000
lr = 5e-3
n_samples = 1
rule = Optimisers.Adam(lr)
state_tree = Optimisers.setup(rule, params_train);  # initialise this optimiser's momentum etc.
losses = []
params_history = []
best_params = copy(params_train)
best_loss = Inf
p = Progress(n_epochs, desc = "Training: ", showspeed = true)
for i in 1:n_epochs
    loss, grads = DifferentiationInterface.value_and_gradient(
        x -> evaluate(x, n_samples), AutoZygote(), params_train)
    state_tree, params_train = Optimisers.update(state_tree, params_train, grads)
    push!(losses, loss)
    push!(params_history, copy(params_train))
    if loss < best_loss
        best_loss = loss
        best_params = copy(params_train)
    end
    update!(p,
        i,
        showvalues = [
            (:epoch, i),
            (:loss, round(loss, digits = 5)),
            (:params, round.(params_train, digits = 5)),
            (:true_params, vcat(true_params...)),
            (:best_loss, round(best_loss, digits = 5))
        ])
end

##
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[1].plot(losses)
ax[2].plot(hcat(params_history...)[1, :], label = "a")
ax[2].plot(hcat(params_history...)[2, :], label = "b")
ax[2].plot(hcat(params_history...)[3, :], label = "theta_low")
ax[2].plot(hcat(params_history...)[4, :], label = "theta_high")
ax[2].axhline(true_params_flat[1], color = "C0")
ax[2].axhline(true_params_flat[2], color = "C1")
ax[2].axhline(true_params_flat[3], color = "C2")
ax[2].axhline(true_params_flat[4], color = "C3")
ax[2].legend()
ax[1].set_yscale("log")
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
    #ax[2].plot(trained_pred[2, :], color = "C0", alpha = 0.5)
    #ax[3].plot(trained_pred[3, :], color = "C0", alpha = 0.5)
    ax[1].plot(prior_pred[1, :], color = "C3", alpha = 0.5)
    #ax[2].plot(prior_pred[2, :], color = "C3", alpha = 0.5)
    #ax[3].plot(prior_pred[3, :], color = "C3", alpha = 0.5)
    #ax[1].plot(true_pred[1, :], color = "C1", alpha = 0.5)
    #ax[2].plot(true_pred[2, :], color = "C1", alpha = 0.5)
    #ax[3].plot(true_pred[3, :], color = "C1", alpha = 0.5)
end
ax[1].set_title("Mean Firm Output")
ax[2].set_title("Mean Agent Effort")
ax[1].plot(data[1, :], color = "black", label = "data")
#ax[2].plot(data[2, :], color = "black", label = "data")
#ax[3].plot(data[3, :], color = "black", label = "data")
ax[1].plot([], [], color = "C0", alpha = 0.5, label = "trained")
ax[1].plot([], [], color = "C1", alpha = 0.5, label = "true")
ax[1].plot([], [], color = "C3", alpha = 0.5, label = "prior")
ax[1].legend()
fig

##
@model function make_ppl_model(data)
    theta_lower ~ Beta(2, 5)
    theta_upper ~ Beta(5, 2)
    initial_effort_lower ~ Beta(2, 5)
    initial_effort_upper ~ Beta(5, 2)
    a ~ Beta(2, 5)
    b ~ Beta(2, 5)
    b = b + 0.5
    params = [theta_lower, theta_upper, initial_effort_lower, initial_effort_upper, a, b]
    data ~ abm(params)
end

function run_vi_with_gradient_method(gradient_method)
    Random.seed!(1)
    prob_model = make_ppl_model(data)
    d = 6
    q = make_masked_affine_autoregressive_flow_torch(d, 16, 32)
    optimizer = Optimisers.AdamW(2e-4, (0.9, 0.99), 1e-5)
    #optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), optimizer)
    q, stats, q_untrained, best_q_cb = run_vi(
        model = prob_model,
        q = q,
        optimizer = optimizer,
        n_montecarlo = 5,
        max_iter = 1000,
        gradient_method = gradient_method,
        adtype = AutoZygote(),
        entropy_estimation = AdvancedVI.MonteCarloEntropy()
    )

    best_q = best_q_cb.best_model
    best_elbo = best_q_cb.best_elbo

    return q, stats, q_untrained, best_q_cb, best_q, best_elbo
end
q_score, stats_score, q_untrained_score, best_q_cb_score, best_q_score, best_elbo_score = run_vi_with_gradient_method("score");
#q_pathwise, stats_pathwise, q_untrained_pathwise, best_q_cb_pathwise, best_q_pathwise, best_elbo_pathwise = run_vi_with_gradient_method("pathwise");
##
println("best_elbo_score $(best_elbo_score)")
println("best_elbo_pathwise $(best_elbo_pathwise)")
elbo_pathwise = [s.elbo for s in stats_pathwise]
elbo_score = [s.elbo for s in stats_score]
fig, ax = plt.subplots(figsize = (6, 4))
ax.plot(-elbo_pathwise, label = "pathwise")
ax.plot(-elbo_score, label = "score")
ax.set_yscale("log")
best_loss = mean([logpdf(abm(true_params_flat), data) for i in 1:20])
ax.axhline(-best_loss, color = "red")
#ax.set_yscale("log")
ax.legend()
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples_score = rand(best_q_score, 10000);
q_samples_score[6, :] = q_samples_score[6, :] .+ 0.5
q_samples_pathwise = rand(best_q_pathwise, 10000);
q_samples_pathwise[6, :] = q_samples_pathwise[6, :] .+ 0.5
q_samples_untrained = rand(q_untrained_pathwise, 10000);
q_samples_untrained[6, :] = q_samples_untrained[6, :] .+ 0.5
prior = Product([Beta(2, 3), Beta(3, 2), Beta(2, 3), Beta(3, 2), Beta(2, 5), Beta(2, 5)])
prior_samples = rand(prior, 10000)
prior_samples[6, :] = prior_samples[6, :] .+ 0.5
fig = pygtc.plotGTC(
    [q_samples_pathwise', q_samples_score', prior_samples'],
    figureSize = 4, truths = vcat(true_params...),
    colorsOrder = ["blues", "oranges", "grays"],
    paramNames = [L"$\theta_a$", L"$\theta_b$", L"$e_a$", L"$e_b$", L"$a$", L"$b$"],
    chainLabels = ["Pathwise", "Score", "Prior"]
)
fig.savefig("../DiffABMsPaper/figures/axtell_posteriors.pdf", bbox_inches = "tight")
fig

##
# predictive
n_samples = 15
q_samples_score_plot = q_samples_score[:, 1:n_samples]
q_samples_pathwise_plot = q_samples_pathwise[:, 1:n_samples]
q_untrained_samples_plot = q_samples_untrained[:, 1:n_samples]
prior_samples_plot = prior_samples[:, 1:n_samples]
fig, ax = plt.subplots(1, figsize = (4, 4))
alpha = 0.5
for i in 1:n_samples
    #q_pred_pw = rand(abm(q_samples_pathwise[:, i]))
    q_pred_pw = rand(abm(q_samples_pathwise_plot[:, i]))
    ax.plot(q_pred_pw[1, :], color = "C0", alpha = alpha)
    #ax[2].plot(q_pred_pw[2, :], color = "C0", alpha = 0.5)
    #ax[3].plot(q_pred_pw[3, :], color = "C0", alpha = 0.5)
    q_pred_score = rand(abm(q_samples_score_plot[:, i]))
    ax.plot(q_pred_score[1, :], color = "C2", alpha = alpha)
    #ax[2].plot(q_pred_score[2, :], color = "C1", alpha = 0.5)
    #ax[3].plot(q_pred_score[3, :], color = "C1", alpha = 0.5)

    #true_pred = rand(abm(true_params_flat))
    #ax[1].plot(true_pred[1, :], color = "C1", alpha = alpha)
    #ax[2].plot(true_pred[2, :], color = "C1", alpha = alpha)
    #ax[3].plot(true_pred[3, :], color = "C1", alpha = alpha)
    prior_pred = rand(abm(prior_samples_plot[:, i]))
    ax.plot(prior_pred[1, :], color = "grey", alpha = alpha, linestyle = "--")
    #ax[2].plot(prior_pred[2, :], color = "C3", alpha = alpha)
    #ax[3].plot(prior_pred[3, :], color = "C3", alpha = alpha)
    #untrained_pred = rand(abm(q_untrained_samples_plot[:, i]))
    #ax[1].plot(untrained_pred[1, :], color = "C2", alpha = alpha)
    #ax[2].plot(untrained_pred[2, :], color = "C2", alpha = alpha)
    #ax[3].plot(untrained_pred[3, :], color = "C2", alpha = alpha)
end
ax.plot(data[1, :], color = "black", label = "Data")
#ax[2].plot(data[2, :], color = "black", label = "data")
#ax[3].plot(data[3, :], color = "black", label = "data")
ax.plot([], [], color = "C0", alpha = 0.5, label = "Pathwise")
ax.plot([], [], color = "C2", alpha = 0.5, label = "Score")
#ax.plot([], [], color = "C2", alpha = 0.5, label = "Untrained")
#ax.plot([], [], color = "C1", alpha = 0.5, label = "true parameters")
ax.plot([], [], color = "grey", alpha = 0.5, label = "Prior", linestyle = "--")
ax.set_xlabel("Timestep")
#ax.set_yscale("log")
ax.legend(title = "Sampled from")
ax.set_title("Mean Firm Output")
fig.savefig("../DiffABMsPaper/figures/axtell_predictions.pdf")
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