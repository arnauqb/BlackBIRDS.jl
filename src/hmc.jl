export run_hmc

function run_hmc(;
        D,
        model,
        initial_p = rand(D),
        n_samples,
        n_adapts,
        δ = 0.65
)
    ℓπ = DynamicPPL.LogDensityFunction(model)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    initial_ϵ = 0.01 #find_good_stepsize(hamiltonian, initial_p)
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = NoAdaptation() #StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(δ, integrator))
    samples, stats = sample(
        hamiltonian, kernel, initial_p, n_samples, adaptor, n_adapts; progress = true)
    return samples, stats
end