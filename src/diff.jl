export value_and_gradient

function value_and_gradient(ad::ADTypes.AbstractADType, model::StochasticModel)
    params, rec = Flux.destructure(model)
    f_to_diff(x) = rand(rec(x))
    value, jacobian = DifferentiationInterface.value_and_jacobian(f_to_diff, ad, params)
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
    for _ in 1:n_samples
        fd = StochasticAD.derivative_estimate(f_aux, params)
        push!(st_samples, hcat(fd...))
    end
    v = rand(model)
    d = sum(st_samples) / n_samples
    return v, d
end

#function value_and_gradient(ad::ADTypes.AutoForwardDiff, model::StochasticModel)
#    function f_aux(x)
#        _, rec_f = Flux.destructure(model)
#        return rand(rec_f(x))[:]
#    end
#    params = Flux.destructure(model)[1]
#    y = zeros(prod(size(model)))
#    result = DiffResults.JacobianResult(y, params)
#    result = ForwardDiff.jacobian!(result, f_aux, params)
#    v, jacobian = DiffResults.value(result), DiffResults.jacobian(result)
#    # restore v to its original shape
#    if length(size(model)) == 2
#        v = Matrix(hcat(Iterators.partition(v, size(model, 2))...)')
#    end
#    return v, jacobian
#end
##
#function value_and_gradient(ad::AutoStochasticAD, model)
#    n_samples = ad.n_samples
#    st_samples = Matrix{Float64}[]
#    params = Flux.destructure(model)[1]
#    function f_aux(x)
#        _, rec_f = Flux.destructure(model)
#        return rand(rec_f(x))[:]
#    end
#    for _ in 1:n_samples
#        fd = StochasticAD.derivative_estimate(f_aux, params)
#        push!(st_samples, hcat(fd...))
#    end
#    v = rand(model)
#    d = sum(st_samples) / n_samples
#    return v, d
#end