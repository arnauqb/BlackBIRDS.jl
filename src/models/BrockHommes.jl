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
    gradient_horizon::Int64
end
@functor BrockHommesModel (p,)

function compute_u(x, g, b, R)
    t1 = x[end] - R * x[end - 1]
    t21 = g .* x[end - 2] .+ b
    t22 = R * x[end - 1]
    t2 = t21 .- t22
    return t1 .* t2
end

function Distributions.rand(bh::BrockHommesModel{T}) where {T}
    beta = 120
    g = [0.0, bh.p[1], bh.p[2], 1.01]
    b = [0.0, bh.p[3], bh.p[4], 0.0]
    sigma = 0.04
    r = 1.0
    R = 1.0 + r
    #g = min.(max.(g, 1e-3), 1.0)

    x = zeros(3)
    res = copy(x)

    for t in 4:bh.n
        # decouple x when hitting time-horizon
        if t % bh.gradient_horizon == 0
            x = drop_gradient(copy(x))
        end
        epsilon = rand(Normal(0.0, 1.0))
        u_h = compute_u(x, g, b, R)
        strategy = Flux.softmax(beta * u_h)
        mean = sum(strategy .* (g .* x[end] + b))
        x_t = (mean + epsilon * sigma) / R
        x = vcat(x, x_t)
        res = vcat(res, x_t)
    end
    return res
end

function Distributions.logpdf(bh::BrockHommesModel{<:Real, <:LLLoss}, y::Vector{Q}) where {Q}
    beta = 120
    g = [0.0, bh.p[1], bh.p[2], 1.01]
    b = [0.0, bh.p[3], bh.p[4], 0.0]
    sigma = 0.04
    r = 1.0
    R = 1.0 + r
    scale = sigma / R
    lp = 0.0
    n_timesteps = bh.n
    for t in 4:n_timesteps
        u_h = compute_u(y[1:t], g, b, R)
        strategy = Flux.softmax(beta * u_h)
        mean = sum(strategy .* (g .* y[t-1] + b)) / R
        lp += logpdf(Normal(mean, scale), y[t])
    end
    return lp
end

end