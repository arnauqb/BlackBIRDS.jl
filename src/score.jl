struct ScoreELBO{EntropyEst <: AdvancedVI.AbstractEntropyEstimator} <:
       AdvancedVI.AbstractVariationalObjective
    entropy::EntropyEst
    n_samples::Int
end

function ScoreELBO(
        n_samples::Int;
        entropy::AdvancedVI.AbstractEntropyEstimator = ClosedFormEntropy()
)
    ScoreELBO(entropy, n_samples)
end

function estimate_energy_with_samples(prob, samples, samples_logprob)
    a = Base.Fix1(LogDensityProblems.logdensity, prob).(AdvancedVI.eachsample(samples))
    return mean(ForwardDiff.value.(a) .* (samples_logprob .- mean(a)))
end

function compute_elbo(q, samples, entropy, obj, problem)
    samples_logprob = logpdf.(Ref(q), AdvancedVI.eachsample(samples)) 
    energy = estimate_energy_with_samples(problem, samples, samples_logprob)
    elbo = energy + entropy
    return elbo
end

function estimate_scoreelbo_ad_forward(params′, aux)
    @unpack rng, obj, problem, restructure, q_stop = aux
    q = restructure(params′)
    samples, entropy = AdvancedVI.reparam_with_entropy(
        rng, q, q_stop, obj.n_samples, obj.entropy)
    elbo = compute_elbo(q, samples, entropy, obj, problem)
    return -elbo
end

function AdvancedVI.estimate_objective(
        rng::Random.AbstractRNG,
        obj::ScoreELBO,
        q,
        prob;
        n_samples::Int = obj.n_samples
)
    samples, entropy = AdvancedVI.reparam_with_entropy(rng, q, q, n_samples, obj.entropy)
    return compute_elbo(q, samples, entropy, obj, prob)
end

function AdvancedVI.estimate_objective(
        obj::ScoreELBO, q, prob; n_samples::Int = obj.n_samples)
    estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

function AdvancedVI.estimate_gradient!(
        rng::Random.AbstractRNG,
        obj::ScoreELBO,
        adtype::ADTypes.AbstractADType,
        out::DiffResults.MutableDiffResult,
        prob,
        params,
        restructure,
        state
)
    q_stop = restructure(params)
    aux = (rng = rng, obj = obj, problem = prob, restructure = restructure, q_stop = q_stop)
    AdvancedVI.value_and_gradient!(
        adtype, estimate_scoreelbo_ad_forward, params, aux, out
    )
    nelbo = DiffResults.value(out)
    stat = (elbo = -nelbo,)
    out, nothing, stat
end
