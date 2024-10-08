using AdvancedVI
using BlackBIRDS
using DynamicPPL

##
@model function max_entropy()
    x ~ filldist(Uniform(0.0, 1.0), 4)
end
d = 4
q = make_masked_affine_autoregressive_flow_torch(d, 4, 16);

model = max_entropy()
optimizer = Optimisers.AdamW(2e-3)

q, stats, q_untrained = run_vi(
	model = model,
	q = q,
	optimizer = optimizer,
	n_montecarlo = 10,
	max_iter = 200,
	gradient_method = "pathwise",
	adtype = AutoZygote(),
	entropy_estimation = AdvancedVI.StickingTheLandingEntropy(),
);

##
elbo_vals = [s.elbo for s in stats];
fig, ax = subplots()
ax.plot(elbo_vals)
fig

##
using PyCall
pygtc = pyimport("pygtc")
q_samples = rand(q, 10000);
q_samples_untrained = rand(q_untrained, 10000);
pygtc.plotGTC([q_samples', q_samples_untrained'],  figureSize = 4, chainLabels = ["trained", "untrained"])
