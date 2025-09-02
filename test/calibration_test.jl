using Test
using AdvancedVI
using BlackBIRDS
using DynamicPPL
using Distributions
using DiffABM
using Zygote
using ForwardDiff
using PyTorchNormalizingFlows
using Optimisers
using Random

@testset "Calibration Test" begin
    # Set seed for reproducibility
    Random.seed!(42)
    
    # Initialize model parameters (same as script)
    rw = DiffABM.RandomWalkParams(100, ST(), [0.2])
    model = ABM(parameters=rw, ad_backend=AutoForwardDiff(), loss=MSELoss(w=1.0))
    
    # Generate data
    y_obs = rand(model)
    
    # Test 1: Check that generated data has correct dimensions
    @test size(y_obs, 1) == 1  # Should have 1 variable
    @test size(y_obs, 2) == 100  # Should have 100 time steps
    
    # Test 2: Check that data values are reasonable for random walk
    @test all(isfinite, y_obs)
    @test !all(y_obs .== y_obs[1])  # Data should vary
    
    # Setup inference model
    @model function inference_model(abm, y_obs)
        p ~ Uniform(0, 1)
        y_obs ~ abm([p])
    end
    
    # Make normalizing flow
    flow = make_masked_affine_autoregressive_flow_torch(dim=1, n_layers=4, n_units=16)
    
    # Test 3: Check flow initialization
    untrained_samples = rand(flow, 100)
    @test size(untrained_samples) == (1, 100)
    @test all(isfinite, untrained_samples)
    
    # Train flow (reduced iterations for test speed)
    flow_trained, stats, flow_untrained, best_model_callback = run_vi(
        model=inference_model(model, y_obs),
        q=flow,
        optimizer=Optimisers.Adam(5e-4),
        n_montecarlo=5,
        max_iter=100,  # Reduced for test speed
        adtype=AutoZygote(),
        gradient_method="pathwise",
        entropy_estimation=AdvancedVI.MonteCarloEntropy(),
    )
    
    # Test 4: Check training stats
    @test length(stats) == 100
    elbo_values = [stat.elbo for stat in stats]
    @test all(isfinite, elbo_values)
    
    # Test 5: Check that ELBO generally improves (allowing some noise)
    # Compare first 5 vs last 5 iterations
    early_elbo = sum(elbo_values[1:5]) / 5
    late_elbo = sum(elbo_values[end-4:end]) / 5
    @test late_elbo > early_elbo - 10.0  # Allow some tolerance for noise
    
    # Test 6: Check trained flow samples
    flow_samples = rand(flow_trained, 1000)
    @test size(flow_samples) == (1, 1000)
    @test all(isfinite, flow_samples)
    
    # Test 7: Check that trained samples are closer to true parameter (0.2)
    sample_mean = sum(flow_samples[1, :]) / length(flow_samples[1, :])
    @test sample_mean ≈ 0.2 atol = 0.1
end