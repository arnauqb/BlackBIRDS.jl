export MSELoss, KDELoss, LLLoss, GaussianMMDLoss

using Distances, LinearAlgebra

abstract type AbstractLoss end

function Distributions.logpdf(d::StochasticModel, y)
    x = rand(d)
    return -d.loss(x, y) / d.loss.w
end

struct LLLoss <: AbstractLoss end

struct MSELoss
    w::Float64
end

(::MSELoss)(x, y) = sum((x - y) .^ 2) / length(y)

struct KDELoss{T, Q} <: AbstractLoss
    n_samples::Int64
    kernel::T
    bandwidth::Q
end
KDELoss(n_samples) = KDELoss(n_samples, Normal, "auto")
KDELoss(n_samples, kernel::Distribution) = KDELoss(n_samples, kernel, "auto")

# Silverman's rule of thumb for KDE bandwidth selection
# taken from KernelDensity.jl
function silverman_rule(data, alpha = 0.9)
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

function Distributions.logpdf(d::StochasticModel{<:KDELoss}, y)
    x_samples = rand(d)
    for _ in 2:(d.loss.n_samples)
        x = rand(d)
        x_samples = hcat(x_samples, x)
    end
    # assume independence estimate kde for each point
    pdf_total = 0.0
    @assert size(x_samples, 1) == length(y)
    for i in axes(x_samples, 1)
        if d.loss.bandwidth == "auto"
            bandwidth = ChainRulesCore.@ignore_derivatives silverman_rule(x_samples[i, :])
        else
            bandwidth = d.loss.bandwidth
        end
        pdf_t = 0.0
        for j in axes(x_samples, 2)
            dist = Normal(x_samples[i, j], bandwidth)
            pdf_t += pdf(dist, y[i])
        end
        pdf_total += pdf_t
    end
    return log(pdf_total / d.loss.n_samples)
end

"""
    GaussianMMDLoss

Shape expected is (n_features, n_timesteps)
"""
struct GaussianMMDLoss{T} <: AbstractLoss
    y::Matrix{T}
    sigma::T
    kernel_yy::Matrix{T}
    w::Float64

    function GaussianMMDLoss(y::AbstractArray{T}, w) where {T}
        if ndims(y) == 1
            y = reshape(y, 1, length(y))
        end
        sigma = estimate_sigma(y)
        kernel_yy = gaussian_kernel(y, y, sigma)
        kernel_yy = kernel_yy - I(size(kernel_yy, 1))
        new{T}(y, sigma, kernel_yy, w)
    end
end

function (loss::GaussianMMDLoss)(x::Matrix, y::Matrix)
    nx = size(x, 1)
    ny = size(y, 1)
    kernel_xy = gaussian_kernel(x, loss.y, loss.sigma)
    kernel_xx = gaussian_kernel(x, x, loss.sigma)
    kernel_xx = kernel_xx - I(size(kernel_xx, 1))
    println("kernel_xx = ", kernel_xx)
    println("kernel_yy = ", loss.kernel_yy)
    println("kernel_xy = ", kernel_xy)
    loss_value = (
        1 / (nx * (nx - 1)) * sum(kernel_xx) +
        1 / (ny * (ny - 1)) * sum(loss.kernel_yy) -
        2 / (nx * ny) * sum(kernel_xy)
    )
    return loss_value
end

function estimate_sigma(y)
    dist = pairwise(SqEuclidean(), y, y, dims = 2)
    # exclude self distances
    mask = I(size(dist, 1))
    dist = dist[.!mask]
    return sqrt(median(dist))
end

function gaussian_kernel(x, y, sigma)
    dist = pairwise(SqEuclidean(), x, y, dims = 2)
    kernel_matrix = @. exp(-(dist) / (2 * sigma^2))
    return kernel_matrix
end

function Distributions.logpdf(
        d::StochasticModel{<:GaussianMMDLoss}, y::AbstractArray{<:Real})
    x = rand(d)
    if ndims(x) == 1
        x = reshape(x, 1, length(x))
    end
    if ndims(y) == 1
        y = reshape(y, 1, length(y))
    end
    return -d.loss(x, y) / d.loss.w
end
