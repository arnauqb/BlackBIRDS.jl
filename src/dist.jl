abstract type Distribution end


struct Normal{T} <: Distribution
    μ::Vector{T}
    σ::Vector{T}
end

struct Beta{T} <: Distribution
    α::Vector{T}
    β::Vector{T}
end

sample(posterior::Distribution, n_samples) = throw("Not implemented")
sample(posterior::PyObject, n_samples) = posterior.sample(n_samples)[1]
sample(posterior::Normal, n_samples) = rand(Distributions.Normal.(posterior.μ, posterior.σ), n_samples)
sample(posterior::Beta, n_samples) = rand(Distributions.Beta.(posterior.α, posterior.β), n_samples)

logpdf(posterior::Distribution, x) = throw("Not implemented")
logpdf(posterior::PyObject, x) = posterior.log_prob(x)
logpdf(posterior::Normal, x) = logpdf(Distributions.Normal.(posterior.μ, posterior.σ), x)

sample_and_logpdf(posterior::PyObject, n_samples) = posterior.sample(n_samples)
sample_and_logpdf(posterior::Distribution, n_samples) = begin
    x = sample(posterior, n_samples)
    log_p = logpdf(posterior, x)
    return x, log_p
end

function compute_kl_divergence(posterior::PyObject, prior::PyObject, n_samples, w)
    x, log_p = sample_and_logpdf(posterior, n_samples)
    log_q = logpdf(prior, x)
    kl_div = (log_p - log_q).mean()
    return w * kl_div
end
function compute_kl_divergence(posterior::Distribution,
        prior, n_samples, w)
    x = sample(posterior, n_samples)
    log_p = logpdf(posterior, x)
    log_q = logpdf(prior, x)
    kl_div = (log_p - log_q).mean()
    return w * kl_div
end

get_kl_values(x) = x
get_kl_values(x::PyObject) = x.data.numpy()[1]
