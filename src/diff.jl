export value_and_gradient

function value_and_gradient(ad::ADTypes.AbstractADType, func, params)
    return DifferentiationInterface.value_and_jacobian(func, ad, params)
end

function value_and_gradient(ad::AutoStochasticAD, func, params)
    n_samples = ad.n_samples
    st_samples = Matrix{Float64}[]
    st_samples = fetch.([Threads.@spawn hcat(StochasticAD.derivative_estimate(
        func, params)...) for _ in 1:n_samples])
    value = func(params)
    jacobian = sum(st_samples) / n_samples
    return value, jacobian
end