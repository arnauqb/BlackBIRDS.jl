export MSELoss, RelativeMSELoss, KDELoss, LLLoss, GaussianMMDLoss, GaussianKernel,
       MMDKernel, CustomLoss, hausdorff_distance, AbstractLoss

using Distances, LinearAlgebra

abstract type AbstractLoss end

struct LLLoss <: AbstractLoss end

struct CustomLoss <: AbstractLoss
    loss::Function
    weights::Vector{Float64}
    w::Float64
    n_samples::Int64
end
function MSELoss(; w=1.0, n_samples=1, weights = [1.0])
    CustomLoss((x, y) -> sum((x - y) .^ 2) / length(y), weights, w, n_samples)
end
function RelativeMSELoss(; w=1.0, n_samples=1, weights = [1.0])
    CustomLoss((x, y) -> sum((x - y) .^ 2 ./ (y .^ 2)) / length(y), weights, w, n_samples)
end
function (loss::CustomLoss)(x, y)
    return loss.loss(x, y)
end

#function Distributions.logpdf(
#        d::StochasticModel{B, L}, y::AbstractVector{<:Real}) where {B, L <: CustomLoss}
#    n_samples = d.loss.n_samples
#    x = [rand(d) for _ in 1:n_samples]
#    return -mean([d.loss(x[i], y) for i in 1:n_samples]) / d.loss.w
#end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractMatrix{<:Real}) where {B, L <: CustomLoss}
    # assume shape is (n_features, n_timesteps)
    n_samples = d.loss.n_samples
    weights = d.loss.weights
    loss = 0.0
    xs = threaded_sampling(d, n_samples)
    #xs = [rand(d) for _ in 1:(d.loss.n_samples)]
    for i in 1:n_samples
        x = xs[i]
        for j in axes(x, 1)
            loss += weights[j] * d.loss(x[j, :], y[j, :])
        end
    end
    return -loss / d.loss.w / size(y, 1) / n_samples
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
    n_samples::Int64
end
GaussianMMDLoss(;w=1.0, n_samples=2) = GaussianMMDLoss(w, n_samples)

function mmd_estimate_sigma(y::AbstractMatrix{T}) where {T}
    sigmas = T[]
    for i in axes(y, 1)
        distances = pairwise(Euclidean(), y[i, :], y[i, :])
        push!(sigmas, median(distances))
    end
    return sigmas
end

function gaussian_kernel(x, y, sigmas)
    covariance = Diagonal(sigmas)
    distances = [norm(x[i, :] - y[i, :]) for i in axes(x, 1)]
    return logpdf(MvNormal(zeros(length(sigmas)), covariance), distances)
end

function Distributions.logpdf(
        d::StochasticModel{B, L}, y::AbstractArray{<:Real, M}) where {
        B, L <: GaussianMMDLoss, M}
    n_samples = d.loss.n_samples
    if n_samples < 2
        throw(ArgumentError("n_samples must be at least 2."))
    end
    # This is broken in Zygote!!
    xs = threaded_sampling(d, n_samples)
    #xs = fetch.([Threads.@spawn rand(d) for _ in 1:(d.loss.n_samples)])
    #xs = [rand(d) for _ in 1:(d.loss.n_samples)]
    sigmas = ChainRulesCore.@ignore_derivatives mmd_estimate_sigma(DiffABM.ignore_gradient.(y))
    iterator = 1:n_samples
    # kxx is the mean of the kernel between each sample except with itself
    kxx = mean([gaussian_kernel(xs[i], xs[j], sigmas)
                for i in iterator, j in iterator if i != j])
    # kyy is the mean of the kernel between each sample and itself
    kyy = mean([gaussian_kernel(xs[i], xs[j], sigmas)
                for i in iterator, j in iterator if i != j])
    # kxy is the mean of the kernel between each sample and the target
    kxy = mean([gaussian_kernel(xs[i], y, sigmas) for i in iterator])
    mmd_loss = kxx + kyy - 2 * kxy
    return -mmd_loss / d.loss.w / size(y, 2)
end
