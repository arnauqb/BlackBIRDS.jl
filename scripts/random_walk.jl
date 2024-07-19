##
import AdvancedVI
using Bijectors
using Flux
using DynamicPPL
using Distributions
using LinearAlgebra
using Optimisers
using Zygote
using CairoMakie

using BlackBIRDS
using BlackBIRDS.RandomWalk

##

n_timesteps = 100
true_p = [0.3]

abm_model = RandomWalkModel(n_timesteps, true_p, MSELoss(1.0));
data = rand(abm_model);

fig = Figure()
lines(fig[1,1], data)
fig

##

@model function ppl_model(data, n_timesteps)
    log_p ~ Normal(0, 0.5)
    p = 10 .^ log_p
    p = clamp(p, 0.0, 1.0)
    data ~ RandomWalkModel(n_timesteps, [p], KDELoss(10))
end

d = 1
q = make_planar_flow(1, 5) #make_masked_affine_autoregressive_flow_torch(d, 4, 16);
#q = AdvancedVI.MeanFieldGaussian(zeros(1), Diagonal(ones(1)));
q_samples_untrained = rand(q, 10^4);
optimizer = Optimisers.AdamW(1e-3);
prob_model = ppl_model(data, n_timesteps);
q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 100,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy(),
);

## plots

using PairPlots

##
elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

##
q_samples = rand(q, 10000)
prior_samples = reshape(rand(Normal(0, 0.5), 10000), 1, 10000);
function make_table(samples)
    return (; log_p = samples[1, :])
end
table = make_table(q_samples);
table_prior = make_table(prior_samples);
table_untrained = make_table(q_samples_untrained);
truths = (; log_p = log10(true_p[1]));
#truths = (; p = log10(true_p[1]))
c1 = Makie.wong_colors(0.5)[1];
c2 = Makie.wong_colors(0.5)[2];
c3 = Makie.wong_colors(0.5)[3];
pairplot(
    PairPlots.Series(table, label="Trained", color=c1, strokecolor=c1),
    PairPlots.Series(table_prior, label="Prior", color=c2, strokecolor=c2),
    PairPlots.Series(table_untrained, label="Untrained", color=c3, strokecolor=c3),
    PairPlots.Truth(truths, color = "black", label="Ground truth"),
)
