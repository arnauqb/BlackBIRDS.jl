##
import AdvancedVI
using Bijectors
using Flux
using DynamicPPL
using DistributionsAD
using Distributions
using LinearAlgebra
using Optimisers
using Zygote
using IPMeasures

using BlackBIRDS
using BlackBIRDS.BrockHommes
using BlackBIRDS.RandomWalk

##

n_timesteps = 100
time_horizon = 1
g2, g3, b2, b3 = 0.9, 0.9, 0.2, -0.2
p = [g2, g3, b2, b3]

abm_model = BrockHommesModel(n_timesteps, p, KDELoss(10), time_horizon);
data = rand(abm_model);


##

@model function ppl_model(data, n)
    p ~ MvNormal([0.5, 0.5, 0.5, -0.5], 0.5)
    data ~ BrockHommesModel(n, p, KDELoss(10), time_horizon)
end

#true_p = [0.25]
#n_timesteps = 100
#abm_model = RandomWalkModel(n_timesteps, true_p, L2Loss());
#
#data = rand(abm_model);
#
#@model function ppl_model(data, n)
#    log_p ~ Normal(0, 1)
#    p = 10 .^ log_p
#    p = clamp(p, 0.0, 1.0)
#    data ~ RandomWalkModel(n, [p], L2Loss())
#end

##
d = 4
#q = make_masked_affine_autoregressive_flow_torch(d, 8, 32, [-1.0 * ones(4), ones(4)]);
q = make_masked_affine_autoregressive_flow_torch(d, 8, 16)
#q = make_planar_flow(4, 20)
q_samples_untrained = rand(q, 10^4);
optimizer = Optimisers.AdamW(1e-3)
prob_model = ppl_model(data, n_timesteps)

q, stats = run_vi(
    model = prob_model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 10,
    max_iter = 200,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy(),
);

## plots

using CairoMakie
using PairPlots

##
elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

##
q_samples = rand(q, 10000)
prior_samples = rand(MvNormal([0.5, 0.5, 0.5, -0.5], 0.5), 10000);
#prior_samples = rand(MvNormal([0.0], 1), 10000);
function make_table(samples)
    #return (; p = samples[1, :])
    return (;
        g2 = samples[1, :],
        g3 = samples[2, :],
        b2 = samples[3, :],
        b3 = samples[4, :],
    )
end
table = make_table(q_samples);
table_prior = make_table(prior_samples);
table_untrained = make_table(q_samples_untrained);
truths = (; g2 = g2, g3 = g3, b2 = b2, b3 = b3);
#truths = (; p = log10(true_p[1]))
c1 = Makie.wong_colors(0.5)[1];
c2 = Makie.wong_colors(0.5)[2];
c3 = Makie.wong_colors(0.5)[3];
pairplot(
    PairPlots.Series(table, label="Trained", color=c1, strokecolor=c1),
    PairPlots.Series(table_prior, label="Prior", color=c2, strokecolor=c2),
    PairPlots.Series(table_untrained, label="Untrained", color=c3, strokecolor=c3),
    PairPlots.Truth(truths, color = "black"),
)