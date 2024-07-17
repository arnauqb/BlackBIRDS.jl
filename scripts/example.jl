##
import AdvancedVI
using Bijectors
using Flux
using DynamicPPL
using DistributionsAD
using Distributions
using LinearAlgebra
using Optimisers

using BlackBIRDS
using BlackBIRDS.RandomWalk

##
n_timesteps = 100
rw_model = RandomWalkModel(n_timesteps, [0.25], KDELoss(10));


data = rand(rw_model);

logpdf(rw_model, data)

@model function ppl_model(data, n)
    log_p ~ Normal(0, 1)
    p = 10 .^ log_p
    p = clamp(p, 0.0, 1.0)
    data ~ RandomWalkModel(n, [p], KDELoss(10))
end
##

d = 1
q = make_planar_flow(d, 20);
q_samples_untrained = rand(q, 10^4)[:];
optimizer = Optimisers.AdamW(1e-3)
prob_model = ppl_model(data, n_timesteps)
q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 500,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy()
);

## plots

using CairoMakie
using PairPlots

##
elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

##
q_samples = rand(q, 10000)[1, :];
prior_samples = rand(Normal(0, 1), 10000);
table = (; log_p = q_samples);
table_prior = (; log_p = prior_samples);
table_untrained = (; log_p = q_samples_untrained);
truths = (; log_p = log10(0.25));
pairplot(table, table_prior, PairPlots.Truth(truths))

