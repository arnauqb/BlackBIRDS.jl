export value_and_gradient

function value_and_gradient(ad::ADTypes.AbstractADType, model::StochasticModel)
    params, rec = Flux.destructure(model)
    f_to_diff(x) = rand(rec(x))
    value, jacobian = DifferentiationInterface.value_and_jacobian(f_to_diff, ad, params)
    # resize to match the model's output
    jacobian = reshape(jacobian, size(model)..., :)
    return value, jacobian
end

function value_and_gradient(ad::AutoStochasticAD, model::StochasticModel)
    n_samples = ad.n_samples
    st_samples = Matrix{Float64}[]
    params = Flux.destructure(model)[1]
    function f_aux(x)
        _, rec_f = Flux.destructure(model)
        return rand(rec_f(x))[:]
    end
    st_samples = fetch.([Threads.@spawn hcat(StochasticAD.derivative_estimate(
        f_aux, params)...)])
    #for _ in 1:n_samples
    #    fd = StochasticAD.derivative_estimate(f_aux, params)
    #    push!(st_samples, hcat(fd...))
    #end
    value = rand(model)
    jacobian = sum(st_samples) / n_samples
    jacobian = reshape(jacobian, size(model)..., :)
    return value, jacobian
end