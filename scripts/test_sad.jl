using StochasticAD
using Distributions
using BlackBIRDS
import DifferentiationInterface

function f(p)
    return rand(Categorical(p))
end

p = rand(100);
p = p ./ sum(p);

values1, jac1 = BlackBIRDS.value_and_gradient(f, DifferentiationInterface.AutoZygote(), p);

values2, jac2 = BlackBIRDS.value_and_gradient(f, AutoStochasticAD(10), p);

