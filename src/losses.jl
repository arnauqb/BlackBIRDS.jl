export L2Loss, KDELoss

abstract type Loss end

function (f::Loss)(model::StochasticModel, y)
    x = rand(model)
    return f(x, y)
end

struct L2Loss <: Loss end

(::L2Loss)(x, y) = sum((x - y).^2)

function Distributions.logpdf(d::StochasticModel{L}, y::Vector{T}) where {L, T}
    x = rand(d)
    return -d.loss(x, y) / length(y)^2
end

struct KDELoss{T, Q} <: Loss
    n_samples::Int64
    kernel::T
    bandwidth::Q
end
KDELoss(n_samples) = KDELoss(n_samples, Normal, "auto")
KDELoss(n_samples, kernel::Distribution) = KDELoss(n_samples, kernel, "auto")

function Distributions.logpdf(d::StochasticModel{<:KDELoss}, y::Vector{T}) where {T}
    x_samples = []
    for _ in 1:d.loss.n_samples
        x = rand(d)
        push!(x_samples, x)
    end
    x_samples = Matrix(hcat(x_samples...)')
    dims = [MultiKDE.ContinuousDim() for _ in 1:length(y)]
    bw = 0.5 * ones(length(y))
    kde = MultiKDE.KDEMulti(dims, bw, x_samples, nothing)
    return log(MultiKDE.pdf(kde, y))
end
