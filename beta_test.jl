using ADTypes
using AdvancedVI
using Bijectors
using BlackBIRDS
using Flux
using DifferentiationInterface
using DynamicPPL
using Optimisers
using Bijectors: Logit;
using Distributions
using Optimisers
using ProgressMeter
using PyPlot
using Random


##
beta_dist = Beta(0.1, 0.1)
samples = rand(beta_dist, 10000)
fig, ax = plt.subplots()
ax.hist(samples, bins = 100, density = true)
fig

function make_flow(dim, nlayers)
    d = MvNormal(zeros(Float32, dim), ones(Float32, dim));
    b = PlanarLayer(dim)
    for i in 2:nlayers
        b = b ∘ PlanarLayer(dim)
    end
    return transformed(d, b)
end


function kl_divergence(q, p)
    values = rand(q, 100)
    lq = [logpdf(q, [v]) for v in values]
    lp = logpdf(p, values)[:]
    return mean(lq .- lp)
end
kl_divergence(q_transformed, Beta(0.1, 0.1))

#q = make_flow(1, 16)
q = make_masked_affine_autoregressive_flow_torch(1, 16, 16);
bijector_transf = inverse(Bijectors.bijector(Beta(0.1, 0.1)))
q_transformed = transformed(q, bijector_transf);
n_epochs = 100
lr = 1e-3
rule = Optimisers.Adam(lr)
v, f = Flux.destructure(q_transformed)
beta_dist = Beta(0.1, 0.1)
state_tree = Optimisers.setup(rule, v);  # initialise this optimiser's momentum etc.
p = Progress(n_epochs, desc = "Training: ", showspeed = true)
function evaluate(v)
    q_transformed = f(v)
    return kl_divergence(q_transformed, beta_dist)
end
loss_values = []
best_v = copy(v)
best_loss = Inf
for i in 1:n_epochs
    loss, grads = DifferentiationInterface.value_and_gradient(
        evaluate, AutoZygote(), v)
    push!(loss_values, loss)
    if loss < best_loss
        best_loss = loss
        best_v = copy(v)
    end
    state_tree, v = Optimisers.update(state_tree, v, grads)
    update!(p,
        i,
        showvalues = [
            (:epoch, i),
            (:loss, round(loss, digits = 5)),
        ])
end

##
fig, ax = plt.subplots()
ax.plot(loss_values)
fig

##
using PyCall
pygtc = pyimport("pygtc")
fit_q_transformed = f(best_v)
q_samples = rand(fit_q_transformed, 10000)
fig = pygtc.plotGTC([q_samples'], figureSize=7)
fig

