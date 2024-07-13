using BlackBIRDS
using PyCall
using Distributions
using Random
using Optimisers
using PyPlot
pygtc = pyimport("pygtc")
torch = pyimport("torch")
nf = pyimport("normflows")

function make_flow(n_parameters)
    K = 8
    flows = []
    for _ in 1:K
        push!(flows, nf.flows.MaskedAffineAutoregressive(n_parameters, 20, num_blocks = 2))
        push!(flows, nf.flows.Permute(n_parameters, mode = "swap"))
    end
    q0 = nf.distributions.DiagGaussian(n_parameters)
    nfm = nf.NormalizingFlow(q0 = q0, flows = flows)
    return nfm
end
function simulator(params::AbstractVector{T}) where {T}
    # random walk
    x = zero(T)
    xs = [zero(T)]
    for _ in 1:100
        p = params[1]
        step = 2 * rand(Bernoulli(p)) - 1
        x += step
        push!(xs, x)
    end
    return xs
end
flow = make_flow(1);
samples_untrained = flow.sample(10000)[1].detach().numpy();

#samples = flow.sample(10000)[1].detach().numpy();
#pygtc.plotGTC([samples], figureSize=10, truths=true_params)
#n_samples = 20
#x_test = [simulator(samples[i, :]) for i in 1:n_samples]
#fig, ax = subplots()
#for i in 1:n_samples
#    ax.plot(x_test[i], color = "C0", alpha = 0.1)
#end
#ax.plot(data, color  = "black")
#fig

true_params = [0.3]
data = simulator(true_params);
#plt.plot(data)

prior = torch.distributions.Normal(-1.0 * torch.ones(1), 0.3 * torch.ones(1))
vi_params = VIParameters(ad_mode = AutoStochasticAD(10), gradient_clipping = 1.0, n_epochs = 200,
    n_samples_per_epoch = 10, n_samples_regularization = 10000, optimiser = AdamW(1e-3), w = 10.0)
loss_params = [2.0]
function loss_fn(model_params, data, loss_params)
    p = clamp.(10 .^ model_params, 0.0, 1.0)
    x = simulator(p)
    return sum((x - data) .^ 2)
end

model_loss, kl_loss, best_params = run_vi!(loss_fn, flow, prior, data, vi_params, loss_params)

BlackBIRDS.update_posterior_estimator!(flow, best_params)



fig, ax = subplots()
ax.plot(model_loss, label = "Model loss")
ax.plot(kl_loss, label = "KL loss")
ax.plot(model_loss + kl_loss, label = "Total loss")
ax.set_yscale("log")
ax.legend()
fig


samples = flow.sample(10000)[1].detach().numpy();
samples_prior = prior.sample([10000]).detach().numpy();
pygtc.plotGTC(
    [samples, samples_untrained, samples_prior], 
    figureSize=10, 
    truths=log10.(true_params), 
    chainLabels=["Posterior", "Untrained", "Prior"])


## plot predictions
n_samples = 20
x_test = [simulator(10 .^ samples[i, :]) for i in 1:n_samples]
fig, ax = subplots()
for i in 1:n_samples
    ax.plot(x_test[i], color = "C0", alpha = 0.3)
end
ax.plot(data, color  = "black")
fig
