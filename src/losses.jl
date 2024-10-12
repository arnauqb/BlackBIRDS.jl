export MSELoss, RelativeMSELoss, KDELoss, LLLoss, GaussianMMDLoss, GaussianKernel, MMDKernel, CustomLoss, hausdorff_distance

using Distances, LinearAlgebra

abstract type AbstractLoss end

struct LLLoss <: AbstractLoss end

struct CustomLoss <: AbstractLoss
    loss::Function
    w::Float64
end
MSELoss(w) = CustomLoss((x, y) -> sum((x - y) .^ 2) / length(y), w)
RelativeMSELoss(w) = CustomLoss((x, y) -> sum((x - y) .^ 2 ./ (y .^ 2)) / length(y), w)
function (loss::CustomLoss)(x, y)
    return loss.loss(x, y)
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractVector{<:Real}) where {B, L <: CustomLoss}
    x = rand(d)
    return -d.loss(x, y) / d.loss.w
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractMatrix{<:Real}) where {B, L <: CustomLoss}
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
        d::StochasticModel{B, L}, y::AbstractMatrix{<:Real}) where {
        B, L <: KDELoss{<:MMDKernel}}
    #x_samples = [rand(d) for _ in 1:(d.loss.n_samples)]
    x_samples = fetch.([Threads.@spawn rand(d) for _ in 1:(d.loss.n_samples)])

    lps = 0.0 # zeros(d.loss.n_samples)
    mmd_loss = GaussianMMDLoss(1.0)
    epsilon = 1e-3
    for i in 1:(d.loss.n_samples)
        loss = mmd_loss(x_samples[i], y)
        lps += exp(-loss^2 / epsilon)
    end
    return log(lps / d.loss.n_samples)
end

"""
    GaussianMMDLoss

Shape expected is (n_features, n_timesteps)
"""
struct GaussianMMDLoss <: AbstractLoss
    w::Float64
end

#function (loss::GaussianMMDLoss)(x, y)
#    if ndims(x) == 1
#        x = reshape(x, 1, length(x))
#    end
#    if ndims(y) == 1
#        y = reshape(y, 1, length(y))
#    end
#    nx = size(x, 2)
#    ny = size(y, 2)
#    sigma = estimate_sigma(y)
#    kernel_xy = gaussian_kernel(x, y, sigma)
#    kernel_xx = gaussian_kernel(x, x, sigma)
#    kernel_xx = kernel_xx - I(size(kernel_xx, 1))
#    kernel_yy = gaussian_kernel(y, y, sigma)
#    kernel_yy = kernel_yy - I(size(kernel_yy, 1))
#    loss_value = (
#        1 / (nx * (nx - 1)) * sum(kernel_xx) +
#        1 / (ny * (ny - 1)) * sum(kernel_yy) -
#        2 / (nx * ny) * sum(kernel_xy)
#    )
#    return loss_value
#end
#
#
#function estimate_sigma(y)
#    dist = pairwise(SqEuclidean(), y, y, dims = 1)
#    # exclude self distances
#    diag = I(size(dist, 1))
#    dist = dist .- dist .* diag
#    return sqrt(non_mutating_median(dist))
#end
#
#function gaussian_kernel(x, y, sigma)
#    dist = pairwise(SqEuclidean(), x, y, dims = 1)
#    kernel_matrix = @. exp(-(dist) / (2 * sigma^2))
#    return kernel_matrix
#end

#function non_mutating_median(arr)
#    sorted = sort(copy(arr), dims = 2)
#    n = size(sorted, 2)
#    return n % 2 == 1 ? sorted[:, (n+1)÷2] : (sorted[:, n÷2] + sorted[:, n÷2+1]) / 2
#end

function mmd_estimate_sigma(X::AbstractMatrix, Y::AbstractMatrix)
    N, T_x = size(X)
    _, T_y = size(Y)
    
    # Combine all data points
    all_points = hcat(X, Y)
    
    # Compute pairwise distances
    n_total = T_x + T_y
    distances = [norm(all_points[:, i] - all_points[:, j]) for i in 1:n_total for j in (i+1):n_total]
    
    # Use median heuristic
    return median(distances)
end

function gaussian_kernel(x::AbstractMatrix, y::AbstractMatrix, sigma::Float64)
    dist_sq = pairwise(SqEuclidean(), x, y, dims=2)
    return exp.(-dist_sq ./ (2 * sigma^2))
end

function (loss::GaussianMMDLoss)(X::AbstractMatrix, Y::AbstractMatrix)
    N, T_x = size(X)
    _, T_y = size(Y)
    
    sigma = ChainRulesCore.@ignore_derivatives mmd_estimate_sigma(DiffABM.ignore_gradient.(X), DiffABM.ignore_gradient.(Y))

    # Compute self-similarity terms
    K_xx = mean(gaussian_kernel(X, X, sigma))
    K_yy = mean(gaussian_kernel(Y, Y, sigma))
    
    # Compute cross-similarity term
    K_xy = mean(gaussian_kernel(X, Y, sigma))
    
    # Compute MMD loss
    mmd = K_xx + K_yy - 2 * K_xy
    
    return mmd
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractArray{<:Real, M}) where {B, L <: GaussianMMDLoss, M}
    x = rand(d)
    return -d.loss(x, y) / d.loss.w
end
