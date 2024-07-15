import AdvancedVI
using DynamicPPL
using Distributions
using LinearAlgebra
using Optimisers

using BlackBIRDS
using BlackBIRDS.RandomWalk

rw_model = RandomWalkModel(100, [0.25], L2Loss());

data = rand(rw_model);


@model function ppl_model(data, n)
    log_p ~ Normal(0, 1)
    p = 10 .^ log_p
    p = clamp(p, 0.0, 1.0)
    data ~ RandomWalkModel(n, [p], L2Loss())
end

d = 1
μ = zeros(d);
L = Diagonal(ones(d));
q = AdvancedVI.MeanFieldGaussian(μ, L)
q_samples_untrained = rand(q, 10^4)[:];
optimizer = Optimisers.Adam(1e-3)
prob_model = ppl_model(data, 100)

q, stats = run_vi(
    model=prob_model,
    q = q,
    optimizer=optimizer,
    n_montecarlo=10,
    max_iter = 10^3,
    adtype = AutoZygote(),
)

## plots

using CairoMakie
using PairPlots

elbo_vals = [s.elbo for s in stats];
plot(elbo_vals)

q_samples = rand(q, 10000)[:];
table = (; log_p = q_samples);
table_untrained = (; log_p = q_samples_untrained);
truths = (; log_p = log10(0.25));
pairplot(table, table_untrained, PairPlots.Truth(truths))
