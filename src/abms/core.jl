export ABM

struct ABM{T, L, H}
    parameters::T
    ad_backend::AbstractADType
    loss::L
    gradient_horizon::H
end
ABM(parameters, ad_backend, loss) = ABM(parameters, ad_backend, loss, Inf)

function Distributions.rand(model::ABM)
    return DiffABM.abm_run(model)
end
Distributions.rand(rng::Random.AbstractRNG, model::ABM) = rand(model)
function Distributions.logpdf(model::ABM{T, L, N}, y) where {L <: AbstractLoss, T, N}
    throw("Loss $L for model $model not implemented")
end