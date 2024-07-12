export VIParameters, run_vi!

struct VIParameters
    ad_mode::ADTypes.AbstractADType
    gradient_clipping::Float64
    n_epochs::Int64
    n_samples_per_epoch::Int64
    n_samples_regularization::Int64
    optimiser::Optimisers.AbstractRule
    w::Float64
end

function VIParameters(; ad_mode, gradient_clipping, n_epochs, n_samples_per_epoch,
        n_samples_regularization, optimiser, w)
    return VIParameters(ad_mode, gradient_clipping, n_epochs, n_samples_per_epoch,
        n_samples_regularization, optimiser, w)
end

function sample(posterior::PyObject, n_samples)
    return posterior.sample(n_samples)[1]
end

function logpdf(posterior::PyObject, x)
    return posterior.log_prob(x)
end

function sample_and_logpdf(posterior::PyObject, n_samples)
    return posterior.sample(n_samples)
end

function compute_kl_divergence(posterior::PyObject, prior, n_samples, w)
    x, log_p = sample_and_logpdf(posterior, n_samples)
    log_q = logpdf(prior, x)
    kl_div = (log_p - log_q).mean()
    return w * kl_div
end

get_kl_values(x) = x
get_kl_values(x::PyObject) = x.data.numpy()[1]

function compute_model_loss(posterior_estimator, loss_fn, data, vi_params, loss_params)
    function loss_aux(params)
        _model_params = params[1:(end - length(loss_params))]
        _loss_params = params[(end - length(loss_params) + 1):end]
        return loss_fn(_model_params, data, _loss_params)
    end
    model_params = sample(posterior_estimator, vi_params.n_samples_per_epoch)
    model_params_values = model_params.detach().numpy()
    total_model_loss = 0.0
    model_jacobians = []
    loss_jacobians = []
    for i in 1:(vi_params.n_samples_per_epoch)
        params = vcat(model_params_values[i, :], loss_params)
        model_loss, jacobian = DifferentiationInterface.value_and_gradient(
            loss_aux, vi_params.ad_mode, params)
        model_jacobian = jacobian[1:(end - length(loss_params))]
        loss_jacobian = jacobian[(end - length(loss_params) + 1):end]
        push!(model_jacobians, model_jacobian)
        push!(loss_jacobians, loss_jacobian)
        total_model_loss += model_loss
    end
    return model_params, total_model_loss / vi_params.n_samples_per_epoch, model_jacobians,
    loss_jacobians
end

function differentiate_model_loss(model_params, model_loss_jacobians, vi_params)
    to_diff = torch.zeros(1)
    for i in 1:(vi_params.n_samples_per_epoch)
        jacobian_torch = torch.tensor(model_loss_jacobians[i], dtype=torch.float32)
        to_diff += torch.matmul(jacobian_torch, py"$model_params[$(i-1), :]")
    end
    to_diff = to_diff / vi_params.n_samples_per_epoch
    to_diff.backward()
end

function differentiate_posterior(posterior_estimator::PyObject, model_params,
        model_loss_jacobians, kl_divergence, vi_params)
    differentiate_model_loss(
        model_params, model_loss_jacobians, vi_params)
    kl_divergence.backward()
    return [p.grad.numpy() for p in posterior_estimator.parameters()]
end

function update_posterior_estimator!(posterior_estimator::PyObject, new_params)
    for (p, new_p) in zip(posterior_estimator.parameters(), new_params)
        p.data = torch.tensor(new_p)
    end
end
update_loss_params!(loss_params, new_params) = loss_params .= new_params

function reset_grads!(posterior_estimator::PyObject)
    for p in posterior_estimator.parameters()
        if p.grad !== nothing
            p.grad.zero_()
        end
    end
end

function get_parameters(posterior_estimator::PyObject)
    [p.data.numpy() for p in posterior_estimator.parameters()]
end

function step_vi!(
        loss_fn, data, posterior_estimator, prior, optim_state, vi_params, loss_params)
    reset_grads!(posterior_estimator)

    full_model = (posterior_estimator = get_parameters(posterior_estimator),
        loss_params = loss_params)

    # compute model loss and jacobians
    model_params, model_loss, model_jacobians, loss_jacobians = compute_model_loss(
        posterior_estimator, loss_fn, data, vi_params, loss_params)

    # compute divergence between posterior and prior
    kl_divergence = compute_kl_divergence(
        posterior_estimator, prior, vi_params.n_samples_regularization, vi_params.w)

    posterior_estimator_grads = differentiate_posterior(
        posterior_estimator, model_params, model_jacobians, kl_divergence, vi_params)
    loss_jacobians = StatsBase.mean(loss_jacobians)
    total_grads = (posterior_estimator = posterior_estimator_grads,
                   loss_params = loss_jacobians)
    optim_state, full_model = Optimisers.update(optim_state, full_model, total_grads)
    update_posterior_estimator!(posterior_estimator, full_model.posterior_estimator)
    update_loss_params!(loss_params, full_model.loss_params)

    return model_loss, get_kl_values(kl_divergence), optim_state
end

function init_vi!(posterior_estimator, vi_params, loss_params)
    full_model = (posterior_estimator = get_parameters(posterior_estimator),
        loss_params = loss_params)
    optim_state = Optimisers.setup(vi_params.optimiser, full_model)
    return optim_state
end

function run_vi!(loss_fn, posterior_estimator, prior, data, vi_params, loss_params)
    optim_state = init_vi!(posterior_estimator, vi_params, loss_params)
    model_loss_hist = Float64[]
    kl_loss_hist = Float64[]
    best_loss = Inf
    best_params = nothing
    for i in 1:vi_params.n_epochs
        model_loss, kl_loss, optim_state = step_vi!(
            loss_fn, data, posterior_estimator, prior, optim_state, vi_params, loss_params)
        total_loss = model_loss + kl_loss
        if total_loss < best_loss
            best_loss = total_loss
            best_params = copy(get_parameters(posterior_estimator))
        end
        println("Epoch: $i, Model loss: $model_loss, KL loss: $kl_loss")
        push!(model_loss_hist, model_loss)
        push!(kl_loss_hist, kl_loss)
    end
    return model_loss_hist, kl_loss_hist, best_params
end
