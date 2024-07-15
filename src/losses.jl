export L2Loss

abstract type Loss end

function (f::Loss)(model::StochasticModel, y)
    x = rand(model)
    return f(x, y)
end

struct L2Loss <: Loss end

(::L2Loss)(x, y) = sum((x - y).^2)

function Distributions.logpdf(d::StochasticModel{L}, y::Vector{T}) where {L, T}
    x = rand(d)
    return -d.loss(x, y)
end
