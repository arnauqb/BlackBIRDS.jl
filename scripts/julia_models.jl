using BlackBIRDS
import Distributions
import DifferentiationInterface
import Flux

struct Normal{T}
    μ::T
    σ::T
end
function sample(d::Normal, n)
    return rand(Distributions.Normal(d.μ, d.σ), n)
end
a = Normal(0.0, 1.0)
v, re = Flux.destructure(a)


DifferentiationInterface.value_and_jacobian(x -> sample(x, 1), DifferentiationInterface.AutoForwardDiff(), a)
