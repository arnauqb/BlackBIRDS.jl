##
import AdvancedVI
using Bijectors
using Flux
using DynamicPPL
using Distributions
using LinearAlgebra
using Optimisers
using Zygote
using PyPlot

using BlackBIRDS
using BlackBIRDS.RandomWalk

##

n_timesteps = 100
true_p = [0.3]

abm_model = RandomWalkModel(n_timesteps, true_p, MSELoss(1.0));
data = rand(abm_model);

fig, ax = subplots()
ax.plot(data)
fig

##

loss = MSELoss(1.0)
@model function ppl_model(data, n_timesteps)
    p ~ Beta(1.0, 1.0)
    data ~ RandomWalkModel(n_timesteps, [p], loss)
end

d = 1
q = make_planar_flow(1, 10) #make_masked_affine_autoregressive_flow_torch(d, 4, 16);
#q = AdvancedVI.MeanFieldGaussian(zeros(1), Diagonal(ones(1)));
optimizer = Optimisers.AdamW(1e-3);
prob_model = ppl_model(data, n_timesteps);
b = inverse(bijector(prob_model));
q_transformed = transformed(q, b);
q_samples_untrained = rand(q_transformed, 10^4);
q, stats = run_vi(
    model = prob_model,
    q = q_transformed,
    optimizer = optimizer,
    n_montecarlo = 20,
    max_iter = 500,
    gradient_method = "pathwise",
    adtype = AutoZygote(),
    entropy_estimation = AdvancedVI.MonteCarloEntropy(),
);

##
elbo_vals = [s.elbo for s in stats];
fig, ax = subplots()
ax.plot(elbo_vals)
fig

## plots
using PyCall
pygtc = pyimport("pygtc")
q_samples = rand(q, 10000)
prior_samples = reshape(rand(Beta(1.0, 1.0), 10000), 1, :);
pygtc.plotGTC([q_samples', prior_samples', q_samples_untrained'],
    figureSize = 7, truths = true_p,
    chainLabels=["trained flow", "prior", "untrained"])

