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

# Silverman's rule of thumb for KDE bandwidth selection
# taken from KernelDensity.jl
function silverman_rule(data, alpha=0.9)
    # Determine length of data
    ndata = length(data)
    ndata <= 1 && return alpha

    # Calculate width using variance and IQR
    var_width = std(data)
    q25, q75 = quantile(data, [0.25, 0.75])
    quantile_width = (q75 - q25) / 1.34

    # Deal with edge cases with 0 IQR or variance
    width = min(var_width, quantile_width)
    if width == 0.0
        if var_width == 0.0
            width = 1.0
        else
            width = var_width
        end
    end

    # Set bandwidth using Silverman's rule of thumb
    return alpha * width * ndata^(-0.2)
end


function Distributions.logpdf(d::StochasticModel{<:KDELoss}, y::Vector{T}) where {T}
    x_samples = rand(d)
    for _ in 2:d.loss.n_samples
        x = rand(d)
        x_samples = hcat(x_samples, x)
    end
    # assume independence estimate kde for each point
    logpdf_total = 0.0
    @assert size(x_samples, 1) == length(y)
    for i in axes(x_samples, 1)
        bandwidth = ChainRulesCore.@ignore_derivatives silverman_rule(x_samples[i, :])
        logpdf_t = 0.0
        for j in axes(x_samples, 2)
            dist = Normal(x_samples[i, j], bandwidth)
            logpdf_t += logpdf(dist, y[j])
        end
        logpdf_total += logpdf_t / d.loss.n_samples / bandwidth
    end
    return logpdf_total
end
