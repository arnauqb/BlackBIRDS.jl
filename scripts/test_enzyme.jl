using BenchmarkTools
using DifferentiationInterface
using ADTypes
using StochasticAD
using Enzyme
Enzyme.API.runtimeActivity!(true)

##
function test(x, y, alpha)
    for alpha_i in alpha
        x = x .+ alpha_i
    end
    return sum(x .* y)
end

x = rand(10000);
y = rand(10000);
alpha = rand(10);

##
@btime DifferentiationInterface.value_and_gradient(
    alpha -> test(x, y, alpha), AutoEnzyme(), alpha);
@btime DifferentiationInterface.value_and_gradient(
    alpha -> test(x, y, alpha), AutoForwardDiff(), alpha);
@btime DifferentiationInterface.value_and_gradient(
    alpha -> test(x, y, alpha), AutoZygote(), alpha);

##
@btime StochasticAD.derivative_estimate(alpha -> test(x, y, alpha), alpha);
@btime StochasticAD.derivative_estimate(alpha -> test(x, y, alpha), alpha,
    StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))));