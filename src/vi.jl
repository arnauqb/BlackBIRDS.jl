export run_vi

#function ChainRulesCore.rrule(
#    ::typeof(AdvancedVI.estimate_energy_with_samples), prob, samples
#)
#    println("hello")
#    fn = Base.Fix1(LogDensityProblems.logdensity, prob)
#    fn_samples =
#        fetch.([
#            Threads.@spawn Zygote.pullback(fn, sample) for
#            sample in AdvancedVI.eachsample(samples)
#        ])
#    values = [sample[1] for sample in fn_samples]
#    pullbacks = [sample[2] for sample in fn_samples]
#    function estimate_energy_with_samples_aux_pullback(ȳ)
#        grads = [pullback(ȳ_i)[1] for (ȳ_i, pullback) in zip(ȳ, pullbacks)]
#        ret = mean(grads)
#        println(size(ret))
#        println(size(samples))
#        return (NoTangent(), NoTangent(), ret)
#    end
#    return mean(values), estimate_energy_with_samples_aux_pullback
#end

mutable struct SaveBestModelCallback{T}
	best_elbo::Float64
	best_model::T
end
function (cb::SaveBestModelCallback)(; stat, state, params, averaged_params, restructure, gradient)
	elbo = stat.elbo
	if elbo > cb.best_elbo
		cb.best_elbo = elbo
		cb.best_model = restructure(params)
	end
	return nothing
end

function run_vi(;
	model,
	q,
	optimizer = Optimisers.Adam(1e-3),
	n_montecarlo,
	max_iter,
	adtype,
	gradient_method = "pathwise",
	entropy_estimation = AdvancedVI.ClosedFormEntropy(),
	transform = "auto",
)
	if transform == "auto"
		bijector_transf = inverse(bijector(model))
		q_transformed = transformed(q, bijector_transf)
	elseif transform === nothing
		q_transformed = q
	else
		bijector_transf = inverse(transform(model))
		q_transformed = transformed(q, bijector_transf)
	end
	ℓπ = DynamicPPL.LogDensityFunction(model)
	if gradient_method == "pathwise"
		elbo = AdvancedVI.RepGradELBO(n_montecarlo, entropy = entropy_estimation)
	elseif gradient_method == "score"
		elbo = AdvancedVI.ScoreGradELBO(n_montecarlo, entropy = entropy_estimation)
	else
		error("Gradient method not recognized")
	end
	q_untrained = deepcopy(q_transformed)
	best_model_callback = SaveBestModelCallback(-Inf, q_transformed)

	q, _, stats, _ = AdvancedVI.optimize(
		ℓπ,
		elbo,
		q_transformed,
		max_iter;
		adtype,
		optimizer = optimizer,
		callback = best_model_callback,
	)
	return q, stats, q_untrained, best_model_callback
end
