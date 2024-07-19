export value_and_gradient

function value_and_gradient(ad::ADTypes.AutoForwardDiff, model::StochasticModel, params)
    function f_aux(x)
        _, rec_f = Flux.destructure(model)
        return rand(rec_f(x))
    end
    y = zeros(size(model))
    result = DiffResults.JacobianResult(y, params)
    result = ForwardDiff.jacobian!(result, f_aux, params)
    v, jacobian = DiffResults.value(result), DiffResults.jacobian(result)
    return v, jacobian
end
#
function value_and_gradient(ad::AutoStochasticAD, model, params)
    n_samples = ad.n_samples
    st_samples = Matrix{Float64}[]
    for _ in 1:n_samples
        fd = StochasticAD.derivative_estimate(f_aux, params)
        push!(st_samples, hcat(fd...))
    end
    v = rand(model)
    d = sum(st_samples) / n_samples
    return v, d
end