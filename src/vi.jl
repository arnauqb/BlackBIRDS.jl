export run_vi

function run_vi(;
        model, 
        q, 
        optimizer = Optimisers.Adam(1e-3),
        n_montecarlo, 
        max_iter, 
        adtype, 
        gradient_method = "pathwise",
        transform = true,
        entropy_estimation = AdvancedVI.ClosedFormEntropy()
    )
    if transform
        bijector_transf = inverse(bijector(model))
        q_transformed = transformed(q, bijector_transf)
    else
        q_transformed = q
    end
    ℓπ = DynamicPPL.LogDensityFunction(model)
    if gradient_method == "pathwise"
        elbo = AdvancedVI.RepGradELBO(n_montecarlo, entropy=entropy_estimation)
    elseif gradient_method == "score"
        elbo = AdvancedVI.ScoreGradELBO(n_montecarlo, entropy=entropy_estimation)
    else
        error("Gradient method not recognized")
    end
    q_untrained = deepcopy(q_transformed)

    q, _, stats, _ = AdvancedVI.optimize(
        ℓπ,
        elbo,
        q_transformed,
        max_iter;
        adtype,
        optimizer = optimizer
    )
    return q, stats, q_untrained
end
