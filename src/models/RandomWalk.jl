module RandomWalk

export RandomWalkModel

using BlackBIRDS
using ChainRulesCore
using Distributions
using Functors
import Random

struct RandomWalkModel{T, L} <: BlackBIRDS.StochasticModel{L}
    n::Int64
    p::Vector{T}
    loss::L
end
@functor RandomWalkModel (p,)

function Distributions.rand(rw::RandomWalkModel{T}) where {T}
    xs = [zero(T)]
    for _ in 2:(rw.n)
        step = 2 * rand(Bernoulli(rw.p[1])) - 1
        x = xs[end] + step
        push!(xs, x)
    end
    return xs
end
Distributions.rand(rng::Random.AbstractRNG, rw::RandomWalkModel) = rand(rw)

function ChainRulesCore.rrule(::typeof(rand), d::RandomWalkModel{T}) where {T}
    v, grad = value_and_gradient(AutoStochasticAD(10), d, d.p)
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        d_tangent = Tangent{RandomWalkModel{T}}(;
            n = NoTangent(), p = grad' * y_tangent, loss = NoTangent())
        return rand_tangent, d_tangent
    end
    return rand(d), rand_pullback
end

end
