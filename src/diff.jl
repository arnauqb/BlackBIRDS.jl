export value_and_gradient

function value_and_gradient(f, mode, x)
    return DifferentiationInterface.value_and_gradient(f, mode, x)
end

function value_and_gradient(f, mode::AutoStochasticAD, x)
    value = f(x)
    gradient = StatsBase.mean([StochasticAD.derivative_estimate(f, x) for _ in 1:mode.n_samples])
    return value, gradient
end
