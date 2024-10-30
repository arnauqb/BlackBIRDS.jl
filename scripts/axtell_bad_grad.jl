using ADTypes
using Bijectors
using Distributions
using DistributionsAD
using ForwardDiff
using LinearAlgebra
using AdvancedVI
using Functors
using DiffABM
using BlackBIRDS
using DiffResults
using Optimisers
using DynamicPPL
using LogDensityProblems
using PyPlot
using Random

##
@functor DiffABM.RandomAxtellAgentInitializer (thetas_bounds, initial_efforts_bounds)
function make_abm_params(thetas_bounds, initial_efforts_bounds, a, b; gradient_horizon = 100)
    n_agents = 1000
    n_timesteps = 40
    neighbors = [sample(1:n_agents, 2, replace = false) for _ in 1:n_agents]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(
        n_agents, thetas_bounds, initial_efforts_bounds, neighbors)
    return DiffABM.AxtellFirmsParams(
        agent_initializer, [a], [b], 0.25, 1.0, n_timesteps, gradient_horizon)
end

@model function make_ppl_model(abm, data)
    theta_lower ~ Uniform(0.0, 0.5)
    theta_upper ~ Uniform(0.5, 1.0)
    a ~ Uniform(0.5, 3.0)
    b ~ Uniform(0.5, 3.0)
    theta_lower = max(0.0, theta_lower)
    theta_upper = min(1.0, theta_upper)
    a = max(0.1, a)
    b = max(0.1, b)
    params = [theta_lower, theta_upper, a, b]
    data ~ abm(params)
end
function estimate_gradient_for_model(ppl_model, objective)
    adtype = ADTypes.AutoForwardDiff()
    problem = DynamicPPL.LogDensityFunction(ppl_model)
    d = 4
    μ = zeros(4) #[0.25, 0.75, 2.0, 2.0]
    L = Diagonal(0.1 * ones(d))
    q = AdvancedVI.MeanFieldGaussian(μ, L)
    bijector_transf = inverse(bijector(ppl_model))
    q = transformed(q, bijector_transf)
    params, restructure = Optimisers.destructure(deepcopy(q))
    state_init = NamedTuple()
    rng = Random.default_rng()
    state = AdvancedVI.maybe_init_objective(
        state_init, rng, objective, problem, params, restructure)
    grad_buf = DiffResults.DiffResult(zero(eltype(params)), similar(params))
    AdvancedVI.estimate_gradient!(
        rng,
        objective,
        adtype,
        grad_buf,
        problem,
        params,
        restructure,
        state
    )
    grad = DiffResults.gradient(grad_buf)
    return grad
end
my_loss = MSELoss(1.0)
abm = ABM(make_abm_params([0.0, 1.0], 1.0, 2.0), AutoForwardDiff(), my_loss)
data = rand(abm)
model = make_ppl_model(abm, data)
objective_rep = RepGradELBO(50)
objective_score = ScoreGradELBO(50)

##
n_samples = 20000
grads_rep = fetch.([Threads.@spawn estimate_gradient_for_model(model, objective_rep)
                    for _ in 1:n_samples])
grads_score = fetch.([Threads.@spawn estimate_gradient_for_model(model, objective_score)
                      for _ in 1:n_samples])

##
fig, axs = plt.subplots(1, 2, figsize = (10, 5), sharex = true)
# take score means as ground truth since it is unbiased
grads_score_nonan = [grads_score[i]
                     for i in 1:length(grads_score)
                     if !any(isinf.(grads_score[i])) && !any(isnan.(grads_score[i]))]
grads_rep_plot = reduce(hcat, grads_rep)'
grads_rep_plot = grads_rep_plot[:, [1, 2, 3, 4, 5, 10, 15, 20]]
grads_score_plot = reduce(hcat, grads_score_nonan)'
grads_score_plot = grads_score_plot[:, [1, 2, 3, 4, 5, 10, 15, 20]]
true_means = mean(grads_score_plot, dims = 1)[:]

axs[1].violinplot(grads_rep_plot, showmeans = true)
axs[1].set_xticks(1:8)
axs[1].set_xticklabels([L"$\mu_1$", L"$\mu_2$", L"$\mu_3$", L"$\mu_4$",
    L"$\sigma_1$", L"$\sigma_2$", L"$\sigma_3$", L"$\sigma_4$"])
axs[1].scatter(
    1:length(true_means), true_means, color = "red", marker = "x", label = "Score")
axs[1].legend()
# filter Inf or Nan from grads_score
axs[2].violinplot(grads_score_plot, showmeans = true)
axs[2].set_xticklabels([L"$\mu_1$", L"$\mu_2$", L"$\mu_3$", L"$\mu_4$",
    L"$\sigma_1$", L"$\sigma_2$", L"$\sigma_3$", L"$\sigma_4$"])
axs[1].set_yscale("symlog")
axs[2].set_yscale("symlog")
fig
