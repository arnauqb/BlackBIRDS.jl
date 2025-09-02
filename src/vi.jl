export run_vi

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

function maximize_flow_entropy(q; n_iters=100, n_samples_per_iter=10, lr=1e-3)
    opt_state = Flux.setup(Flux.Adam(lr), q)
    p = Progress(n_iters, desc="Maximizing flow entropy...")
    for i in 1:n_iters
        loss, grads = Flux.withgradient(q) do q
            entropy = 0.0
            for j in 1:n_samples_per_iter
                sample = rand(q)
                entropy += -logpdf(q, sample)
            end
            -entropy / n_samples_per_iter
        end
        Flux.update!(opt_state, q, grads[1])
        next!(p, showvalues=[(:entropy, -loss)])
    end
    return q
end

function run_vi(;
    model,
    q,
    optimizer=Optimisers.Adam(1e-3),
    n_montecarlo,
    max_iter,
    adtype,
    gradient_method="pathwise",
    entropy_estimation=AdvancedVI.ClosedFormEntropy(),
    transform="auto",
    maximize_initial_entropy=false,
    init_entropy_n_iters=100,
    init_entropy_n_samples_per_iter=10,
    init_entropy_lr=1e-3,
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
    if maximize_initial_entropy
        q_transformed = maximize_flow_entropy(
            q_transformed;
            n_iters=init_entropy_n_iters,
            n_samples_per_iter=init_entropy_n_samples_per_iter,
            lr=init_entropy_lr
        )
    end
    ℓπ = DynamicPPL.LogDensityFunction(model)
    if gradient_method == "pathwise"
        elbo = AdvancedVI.RepGradELBO(n_montecarlo, entropy=entropy_estimation)
    elseif gradient_method == "score"
        elbo = AdvancedVI.ScoreGradELBO(n_montecarlo)
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
        optimizer=optimizer,
        callback=best_model_callback,
    )
    return q, stats, q_untrained, best_model_callback
end
