export ABM

struct ABM{P, B, L, H} <: StochasticModel{B, L}
    parameters::P
    ad_backend::B
    loss::L
    gradient_horizon::H
end
function (abm::ABM)(params)
    _, rec_f = Flux.destructure(abm)
    return rec_f(params)
end
Functors.@functor ABM (parameters,)
#Flux.@layer ABM trainable=(parameters,)
ABM(parameters, ad_backend, loss) = ABM(parameters, ad_backend, loss, Inf)

function Distributions.rand(model::ABM)
    params, rec_f = Flux.destructure(model)
    return diff_rand(model.ad_backend, rec_f, params)
end
Distributions.rand(rng::Random.AbstractRNG, model::ABM) = rand(model)
#function Distributions.logpdf(model::ABM{P, B, L, H}, y) where {P, B, L <: AbstractLoss, H}
#    throw("Loss $L for model $model not implemented")
#end

## differentiation rules

function diff_rand(ad_backend, rec_f, params)
    return abm_run(rec_f(params).parameters)
end

function ChainRulesCore.rrule(
        ::typeof(diff_rand), ad::AutoForwardDiff, rec_f, params)
    value, jacobian = DifferentiationInterface.value_and_jacobian(
        x -> abm_run(rec_f(x).parameters), ad, params)
    function diff_rand_pullback(y_tangent)
        grad = jacobian' * y_tangent[:]
        return NoTangent(), NoTangent(), NoTangent(), grad
    end
    return value, diff_rand_pullback
end
function ChainRulesCore.rrule(
        ::typeof(diff_rand), ad::AutoStochasticAD, rec_f, params)
    n_samples = ad.n_samples
    st_samples = Matrix{Float64}[]
    st_samples = fetch.([Threads.@spawn hcat(StochasticAD.derivative_estimate(
                             x -> abm_run(rec_f(x).parameters), params)...) for _ in 1:n_samples])
    #st_samples = [hcat(StochasticAD.derivative_estimate(
    #                  x -> abm_run(rec_f(x).parameters), params)...) for _ in 1:n_samples]
    value = abm_run(rec_f(params).parameters)
    jacobian = sum(st_samples) / n_samples
    function diff_rand_pullback(y_tangent)
        return NoTangent(), NoTangent(), NoTangent(), jacobian' * y_tangent[:]
    end
    return value, diff_rand_pullback
end

#non_zygote_backends = Union{AutoStochasticAD, AutoForwardDiff}
#function ChainRulesCore.rrule(
#        ::typeof(rand), model::ABM{P, B, L, H}) where {P, B <: non_zygote_backends, L, H}
#    v, jacobians = value_and_gradient(model.ad_backend, model)
#    function rand_pullback(y_tangent)
#        rand_tangent = NoTangent()
#        grad = jacobians' * y_tangent
#        p_grads = Dict{Symbol, Vector{eltype(grad[1])}}()
#        counter = 1
#        for (key, p) in pairs(Flux.trainable(model.parameters))
#            p_grads[key] = grad[counter:(counter + length(p) - 1)]
#            counter += length(p)
#        end
#        params_tangent = Tangent{P}(; p_grads...)
#        model_tangent = Tangent{ABM{P, B, L, H}}(; parameters = params_tangent)
#        return rand_tangent, model_tangent
#    end
#    return v, rand_pullback
#end