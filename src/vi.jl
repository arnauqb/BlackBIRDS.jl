export run_vi

function run_vi(;
        model, 
        q, 
        optimizer = Optimisers.Adam(1e-3),
        n_montecarlo, 
        max_iter, 
        adtype, 
        gradient_method = "pathwise",
        entropy_estimation = AdvancedVI.ClosedFormEntropy()
    )
    ℓ = DynamicPPL.LogDensityFunction(model)
    if gradient_method == "pathwise"
        elbo = AdvancedVI.RepGradELBO(n_montecarlo, entropy=entropy_estimation)
    elseif gradient_method == "score"
        elbo = ScoreELBO(n_montecarlo)
    else
        error("Gradient method not recognized")
    end

    q, stats, _ = AdvancedVI.optimize(
        ℓ,
        elbo,
        q,
        max_iter;
        adtype,
        optimizer = optimizer
    )
    return q, stats
end
