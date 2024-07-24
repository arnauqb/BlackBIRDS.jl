export ABM

struct ABM{P, B, L, H} <: StochasticModel{B, L}
    parameters::P
    ad_backend::B
    loss::L
    gradient_horizon::H
end
Functors.@functor ABM (parameters, )
ABM(parameters, ad_backend, loss) = ABM(parameters, ad_backend, loss, Inf)

function Distributions.rand(model::ABM)
    return DiffABM.abm_run(model.parameters)
end
Distributions.rand(rng::Random.AbstractRNG, model::ABM) = rand(model)
function Distributions.logpdf(model::ABM{P, B, L, H}, y) where {P, B, L <: AbstractLoss, H}
    throw("Loss $L for model $model not implemented")
end

## differentiation rules

non_zygote_backends = Union{AutoStochasticAD, AutoForwardDiff}
function ChainRulesCore.rrule(
        ::typeof(rand), model::ABM{P, B, L, H}) where {P, B<:non_zygote_backends, L, H}
    v, jacobians = value_and_gradient(model.ad_backend, model)
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        grad = jacobians' * y_tangent
        p_grads = Dict{Symbol, Vector{eltype(grad[1])}}()
        counter = 1
        for (key, p) in pairs(Flux.trainable(model.parameters))
            p_grads[key] = grad[counter:(counter + length(p) - 1)]
            counter += length(p)
        end
        params_tangent = Tangent{P}(;p_grads...)
        model_tangent = Tangent{ABM{P, B, L, H}}(; parameters=params_tangent)
        return rand_tangent, model_tangent
    end
    return v, rand_pullback
end