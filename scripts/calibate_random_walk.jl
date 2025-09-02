using AdvancedVI
using BlackBIRDS
using DynamicPPL
using Distributions
using DiffABM
using PyPlot
using Zygote
using ForwardDiff
#using PyTorchNormalizingFlows
using Optimisers


## init model parameters
rw = DiffABM.RandomWalkParams(100, ST(), [0.2]) # Use Straight-Through Estimator

# use autoforwarddiff to diff, use MSELoss as likelihood 
model = ABM(parameters=rw, ad_backend=AutoZygote(), loss=MSELoss(w=2.0))

# generate data
y_obs = rand(model)

fig, ax = plt.subplots()
ax.plot(y_obs[1,:])
fig

## setup inference problem

@model function inference_model(abm, y_obs)
    p ~ Uniform(0, 1)
    y_obs ~ abm([p])
end

## make normalizing flow
flow = make_masked_affine_autoregressive_flow_torch(dim=1, n_layers=4, n_units=16);


## train flow
flow_trained, stats, flow_untrained, best_model_callback = run_vi(
    model=inference_model(model, y_obs),
    q=flow,
    optimizer=Optimisers.Adam(5e-4),
    n_montecarlo=5,
    max_iter=100,
    adtype=AutoZygote(), # use zygote for flow, Forwarddiff for abm
    gradient_method="pathwise",
    entropy_estimation=AdvancedVI.MonteCarloEntropy(),
);

## plot elbo
elbo_values = [stat.elbo for stat in stats]
fig, ax = plt.subplots()
ax.plot(elbo_values)
ax.set_title("ELBO")
ax.set_xlabel("Iteration")
ax.set_ylabel("ELBO")
fig.savefig("img/elbo.png")
fig

##
flow_samples = rand(flow_trained, 5000)
untrained_samples = rand(flow_untrained, 5000)

fig, ax = plt.subplots()
ax.hist(flow_samples[1,:], label="trained", alpha=0.5, bins=100)
ax.hist(untrained_samples[1,:], label="untrained", alpha=0.5, bins=100)
ax.legend()
ax.axvline(0.2, color="red")
fig.savefig("img/flow_samples.png")
fig

