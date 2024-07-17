module BrockHommes

export BrockHommesModel

using BlackBIRDS
using Distributions
using Flux
using Functors

struct BrockHommesModel{T, L} <: BlackBIRDS.StochasticModel{L}
    n::Int64
    p::Vector{T}
    loss::L
end
@functor BrockHommesModel (p,)

function Distributions.rand(bh::BrockHommesModel{T}) where {T}
    beta = 120
    g = [0.0, bh.p[1], bh.p[2], 0.0]
    b = [0.0, bh.p[3], bh.p[4], 0.0]
    sigma = 0.04
    r = 1.0
    R = 1.0 + r
    g = min.(max.(g, 1e-3), 1.0)

    x = zeros(3)

    for _ in 1:bh.n
        epsilon = rand(Normal(0.0, 1.0))
        exponent = @. beta * (x[end] - R * x[end - 1]) *
                      (g * x[end - 2] + b - R * x[end - 1])
        norm_exponentiated = Flux.softmax(exponent)
        mean = sum(norm_exponentiated .* (g .* x[end] + b))
        x_t = (mean + epsilon + sigma) / R
        x = vcat(x, x_t)
    end
    return x[4:end]
end

end