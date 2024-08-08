export MSELoss, KDELoss, LLLoss, GaussianMMDLoss, GaussianKernel, MMDKernel

using Distances, LinearAlgebra

abstract type AbstractLoss end

struct LLLoss <: AbstractLoss end

struct MSELoss
    w::Float64
end

(::MSELoss)(x, y) = sum((x - y) .^ 2) / length(y)

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractVector{<:Real}) where {B, L <: MSELoss}
    x = rand(d)
    return -d.loss(x, y) / d.loss.w
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractMatrix{<:Real}) where {B, L <: MSELoss}
    # assume shape is (n_features, n_timesteps)
    x = rand(d)
    loss = 0.0
    for i in axes(x, 1)
        loss += d.loss(x[i, :], y[i, :])
    end
    return -loss / d.loss.w / size(x, 1)
end

abstract type KDEKernel end

struct GaussianKernel{T} <: KDEKernel
    bw::T
end
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

struct MMDKernel <: KDEKernel end

struct KDELoss{T} <: AbstractLoss
    n_samples::Int64
    kernel::T
end

function compute_kde_loss(kernel::GaussianKernel, x_samples, y)
    logpdf_total = 0.0
    @assert size(x_samples[:, 1]) == size(y)
    for i in axes(x_samples, 1)
        if kernel.bw == "auto"
            bandwidth = ChainRulesCore.@ignore_derivatives silverman_rule(x_samples[i, :])
        else
            bandwidth = kernel.bw
        end
        pdf_t = 0.0
        for j in axes(x_samples, 2)
            dist = Normal(x_samples[i, j], bandwidth)
            pdf_t += pdf(dist, y[i])
        end
        logpdf_total += log(pdf_t / size(x_samples, 2) + 1e-8)
    end
    return logpdf_total
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractVector{<:Real}) where {B, L <: KDELoss}
    x_samples = fetch.([Threads.@spawn rand(d) for _ in 1:(d.loss.n_samples)])
    x_samples = reduce(hcat, x_samples)
    return compute_kde_loss(d.loss.kernel, x_samples, y)
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractMatrix{<:Real}) where {B, L <: KDELoss}
    x_samples = fetch.([Threads.@spawn rand(d) for _ in 1:(d.loss.n_samples)])
    x_samples = cat(x_samples..., dims = 3)
    logpdf_total = 0.0
    for i in axes(x_samples, 1)
        logpdf_total += compute_kde_loss(d.loss.kernel, x_samples[i, :, :], y[i, :])
    end
    return logpdf_total
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractVector{<:Real}) where {
        B, L <: KDELoss{<:MMDKernel}}
    x_samples = [rand(d) for _ in 1:(d.loss.n_samples)]
    #x_samples = fetch.([Threads.@spawn rand(d) for _ in 1:(d.loss.n_samples)])

    x_samples = hcat(x_samples...)
    lps = 0.0 # zeros(d.loss.n_samples)
    mmd_loss = GaussianMMDLoss(y, 1.0)
    epsilon = 1e-3
    for i in 1:(d.loss.n_samples)
        x = x_samples[:, i]
        loss = mmd_loss(x, y)
        lps += exp(-loss^2 / epsilon)
    end
    return log(lps / d.loss.n_samples)
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
        sigma = ChainRulesCore.@ignore_derivatives estimate_sigma(y)
        kernel_yy = gaussian_kernel(y, y, sigma)
        kernel_yy = kernel_yy - I(size(kernel_yy, 1))
        new{T}(y, sigma, kernel_yy, w)
    end
end

function (loss::GaussianMMDLoss)(x, y)
    if ndims(x) == 1
        x = reshape(x, 1, length(x))
    end
    if ndims(y) == 1
        y = reshape(y, 1, length(y))
    end
    nx = size(x, 2)
    ny = size(y, 2)
    kernel_xy = gaussian_kernel(x, loss.y, loss.sigma)
    kernel_xx = gaussian_kernel(x, x, loss.sigma)
    kernel_xx = kernel_xx - I(size(kernel_xx, 1))
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
    diag = I(size(dist, 1))
    dist = dist .- dist .* diag
    return sqrt(median(dist))
end

function gaussian_kernel(x, y, sigma)
    dist = pairwise(SqEuclidean(), x, y, dims = 2)
    kernel_matrix = @. exp(-(dist) / (2 * sigma^2))
    return kernel_matrix
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractArray{<:Real}) where {B, L <: GaussianMMDLoss}
    x = rand(d)
    return -d.loss(x, y) / d.loss.w
end
