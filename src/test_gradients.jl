export test_model_gradients

using FiniteDifferences
using StatsBase

function get_average(model::StochasticModel, x; n_samples)
    _, rec_f = Flux.destructure(model)
    model = rec_f(x)
    return sum(fetch.([Threads.@spawn rand(model) for _ in 1:n_samples])) / n_samples
end

function test_model_gradients(
        model, ad, x; n_samples_ad, n_samples_fd, fdm, atol = 1e-2, rtol = 1e-2)
    _, rec_f = Flux.destructure(model)
    model = rec_f(x)
    results = fetch.([Threads.@spawn value_and_gradient(ad, model) for _ in 1:n_samples_ad])
    values_ad = [result[1] for result in results]
    jacobians_ad = [result[2] for result in results]
    value_fd, jacobian_fd = DifferentiationInterface.value_and_jacobian(
        x -> get_average(model, x, n_samples = n_samples_fd), AutoFiniteDifferences(fdm = fdm), x)
    value_ad = sum(values_ad) / n_samples_ad
    jacobian_ad = sum(jacobians_ad) / n_samples_ad
    if !all(isapprox.(value_ad, value_fd, atol = atol, rtol = rtol))
        println("Values do not match")
    else
        println("Values match!")
    end
    if !all(isapprox.(jacobian_ad, jacobian_fd, atol = atol, rtol = rtol))
        println("Jacobians do not match")
    else
        println("Jacobians match!")
    end
    return values_ad, jacobians_ad, value_fd, jacobian_fd
end