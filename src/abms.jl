export ABM

struct ABM{P,B,L,H,S} <: StochasticModel{B,L}
    parameters::P
    ad_backend::B
    loss::L
    gradient_horizon::H
    summarizer::S
end
function ABM(; parameters, ad_backend, loss, summarizer=x -> x, gradient_horizon=Inf)
    ABM(parameters, ad_backend, loss, gradient_horizon, summarizer)
end
function (abm::ABM)(params)
    _, rec_f = Flux.destructure(abm)
    return rec_f(params)
end
Functors.@functor ABM (parameters,)

function Distributions.rand(model::ABM)
    params, rec_f = Flux.destructure(model)
    return diff_rand(model.ad_backend, rec_f, params)
end
Distributions.rand(rng::Random.AbstractRNG, model::ABM) = rand(model)

## differentiation rules

function diff_rand(ad_backend, rec_f, params)
    model = rec_f(params)
    x = abm_run(model.parameters)
    return model.summarizer(x)
end

function ChainRulesCore.rrule(
    ::typeof(diff_rand), ad::AutoForwardDiff, rec_f, params)
    value, jacobian = DifferentiationInterface.value_and_jacobian(
        x -> diff_rand(ad, rec_f, x), ad, params)
    n_params = length(params)
    #jacobian = reshape(jacobian, n_params, :)'
    function diff_rand_pullback(y_tangent)
        grad = jacobian' * y_tangent[:]
        return NoTangent(), NoTangent(), NoTangent(), grad
    end
    return value, diff_rand_pullback
end

function value_and_jacobian(f, ad::AutoStochasticAD, params)
    n_samples = ad.n_samples
    samples = [hcat(StochasticAD.derivative_estimate(x -> f(x)[:], params)...) for _ in 1:n_samples]
    jacobian = sum(samples) / n_samples
    f_value = f(params)
    return f_value, jacobian
end
function ChainRulesCore.rrule(
    ::typeof(diff_rand), ad::AutoStochasticAD, rec_f, params)
    value, jacobian = value_and_jacobian(x -> diff_rand(ad, rec_f, x), ad, params)
    function diff_rand_pullback(y_tangent)
        grad = jacobian' * y_tangent[:]
        return NoTangent(), NoTangent(), NoTangent(), grad
    end
    return value, diff_rand_pullback
end