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
function make_abm_params(a, b)
    n_agents = 1000
    n_timesteps = 100
    neighbors = [sample(1:n_agents, rand(2:6), replace = false) for _ in 1:n_agents]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(
        n_agents, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), neighbors)
    return DiffABM.AxtellFirmsParams(
        agent_initializer, [a], [b], 0.05, 1.0, n_timesteps, 1.0)
end
true_params = [0.5, 2.0]
prior = Product([Uniform(0.0, 5.0), Uniform(0.0, 5.0)])
params_to_try = rand(prior, 5)
abm = ABM(make_abm_params(true_params...), AutoForwardDiff(), KDELoss(10, MMDKernel()))
data = rand(abm(true_params))
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for i in 1:size(params_to_try, 2)
    toplot = rand(abm(params_to_try[:, i]))
    ax[1].plot(toplot[1, :],
        label = "a=$(round(params_to_try[1, i], digits=2)), b=$(round(params_to_try[2, i], digits=2))")
    ax[2].plot(toplot[2, :],
        label = "a=$(round(params_to_try[1, i], digits=2)), b=$(round(params_to_try[2, i], digits=2))")
    ax[3].plot(toplot[3, :],
        label = "a=$(round(params_to_try[1, i], digits=2)), b=$(round(params_to_try[2, i], digits=2))")
end
ax[1].plot(data[1, :], color = "black", label = "data")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[1].legend()
fig

##
@model function make_ppl_model(data)
    a ~ Uniform(0.0, 5.0)
    b ~ Uniform(0.0, 5.0)
    data ~ abm([a, b])
end

prob_model = make_ppl_model(data)
##
d = 2
q = make_masked_affine_autoregressive_flow_torch(2, 16, 32);
optimizer = Optimisers.AdamW(1e-4, (0.9, 0.99), 1e-4)
#optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(5.0), optimizer)
q, stats, q_untrained, best_q_cb = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 200,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    #entropy_estimation = AdvancedVI.MonteCarloEntropy(),
    entropy_estimation = AdvancedVI.StickingTheLandingEntropy()
);
best_q = best_q_cb.best_model
best_elbo = best_q_cb.best_elbo

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
prior = Product([Uniform(0.0, 5.0), Uniform(0.0, 5.0)])
prior_samples = rand(prior, 10000)
fig = pygtc.plotGTC([q_samples', q_samples_untrained', prior_samples'],
    figureSize = 8, truths = true_params,
    paramNames = ["a", "b"], chainLabels = ["trained flow", "untrained flow", "prior"])
#fig.savefig("figures/axtell_posteriors.pdf")

##
# predictive
n_samples = 15
q_samples = rand(best_q, n_samples);
q_untrained_samples = rand(q_untrained, n_samples);
prior_samples = rand(prior, n_samples)
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
alpha = 0.25
for i in 1:n_samples
    q_pred = rand(abm([q_samples[:, i]...]))
    ax[1].plot(q_pred[1, :], color = "C0", alpha = 0.5)
    ax[2].plot(q_pred[2, :], color = "C0", alpha = 0.5)
    ax[3].plot(q_pred[3, :], color = "C0", alpha = 0.5)
    true_pred = rand(abm([true_params[i] for i in 1:length(true_params)]))
    #ax[1].plot(true_pred[1, :], color = "C1", alpha = alpha)
    #ax[2].plot(true_pred[2, :], color = "C1", alpha = alpha)
    #ax[3].plot(true_pred[3, :], color = "C1", alpha = alpha)
    prior_pred = rand(abm([prior_samples[:, i]...]))
    ax[1].plot(prior_pred[1, :], color = "C3", alpha = alpha)
    ax[2].plot(prior_pred[2, :], color = "C3", alpha = alpha)
    ax[3].plot(prior_pred[3, :], color = "C3", alpha = alpha)
end
ax[1].plot(data[1, :], color = "black", label = "data")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[2].plot([], [], color = "C0", alpha = 0.5, label = "trained flow")
ax[2].plot([], [], color = "C1", alpha = 0.5, label = "true parameters")
ax[3].plot([], [], color = "C3", alpha = 0.5, label = "prior")
for i in 1:3
    ax[i].set_xlabel("Timestep")
end
ax[2].legend(loc = "center left", bbox_to_anchor = (1, 0.5))
ax[1].set_ylabel("Mean Agent Effort")
ax[2].set_ylabel("Mean Firm Output")
ax[3].set_ylabel("Mean Firm Size")
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