using ADTypes
using BlackBIRDS
using DiffABM
using Flux
using Test
using Zygote

sad_backend_samplers = [
    (AutoStochasticAD(10), SAD())
]
forward_backends_samplers = [
    (AutoForwardDiff(), SM()),
    (AutoForwardDiff(), ST()),
    (AutoForwardDiff(), GS(0.1))
]
reverse_backends_samplers = [
    (AutoZygote(), SM()),
    (AutoZygote(), ST()),
    (AutoZygote(), GS(0.1))
]
all_samplers = vcat(
    sad_backend_samplers, forward_backends_samplers, reverse_backends_samplers)
loss = MSELoss(1.0)

@testset "test Game of Life" begin
    for (backend, discrete_sampler) in vcat(sad_backend_samplers, forward_backends_samplers)
        params = GameOfLifeParams(100, 10, rand(10), [0.1], discrete_sampler)
        abm = ABM(params, backend, loss)
        v, f = Zygote.pullback(rand, abm)
        grad = f(v)
        @test length(grad[1].parameters.probs) == 10
        @test length(grad[1].parameters.initial_prob) == 1
    end
end

@testset "test Random Walk" begin
    for (backend, discrete_sampler) in all_samplers
        params = RandomWalkParams(100, discrete_sampler, [0.1])
        abm = ABM(params, backend, loss)
        v, f = Zygote.pullback(rand, abm)
        grad = f(v)
        @test length(grad[1].parameters.p) == 1
    end
end

@testset "test Brock Hommes" begin
    p = rand(11)
    for (backend, discrete_sampler) in vcat(
        forward_backends_samplers, reverse_backends_samplers)
        params = BrockHommesParams(100, p)
        abm = ABM(params, backend, loss)
        v, f = Zygote.pullback(rand, abm)
        grad = f(v)
        @test length(grad[1].parameters.params) == 11
    end
end

@testset "test SIR" begin
    n_agents = 100
    venues = Symbol.(["household", "company", "school", "leisure"])
    fraction_population_per_venue = [1.0, 0.4, 0.4, 1.0]
    number_per_venue = [1, 2, 3, 4]
    graph = DiffABM.generate_random_world_graph(
        n_agents, venues, fraction_population_per_venue, number_per_venue)
    initial_infected = [0.1]
    gamma = [0.05]
    betas = [0.5, 0.2, 0.3, 0.1]
    n_timesteps = 30
    delta_t = 1.0
    infection_type = ConstantInfection()
    policies = Policies()
    for (backend, discrete_sampler) in all_samplers
        params = SIRParams(
            graph, initial_infected, betas, venues, gamma, delta_t, n_timesteps,
            discrete_sampler, infection_type, policies)
        abm = ABM(params, backend, loss)
        v, f = Zygote.pullback(rand, abm)
        grad = f(v)
        @test length(grad[1].parameters.initial_infected) == 1
        @test length(grad[1].parameters.initial_infected) == 1
        @test length(grad[1].parameters.venue_betas) == 4
    end
end
