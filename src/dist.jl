
function sample(posterior::PyObject, n_samples)
    return posterior.sample(n_samples)[1]
end
sample(posterior::Distributions.Distribution, n_samples) = rand(posterior, n_samples)

function logpdf(posterior::PyObject, x)
    return posterior.log_prob(x)
end
logpdf(posterior::Distributions.Distribution, x) = logpdf(posterior, x)

function sample_and_logpdf(posterior::PyObject, n_samples)
    return posterior.sample(n_samples)
end
function sample_and_logpdf(posterior::Distributions.Distribution, n_samples)
    sample_and_logpdf(posterior, n_samples)
end

function compute_kl_divergence(posterior::PyObject, prior::PyObject, n_samples, w)
    x, log_p = sample_and_logpdf(posterior, n_samples)
    log_q = logpdf(prior, x)
    kl_div = (log_p - log_q).mean()
    return w * kl_div
end
function compute_kl_divergence(posterior::Distributions.Distribution,
        prior::Distributions.Distribution, n_samples, w)
    x = sample(posterior, n_samples)
    log_p = logpdf(posterior, x)
    log_q = logpdf(prior, x)
    kl_div = (log_p - log_q).mean()
    return w * kl_div
end

get_kl_values(x) = x
get_kl_values(x::PyObject) = x.data.numpy()[1]
