module SIRJune

export SIRJuneModel, generate_random_world_graph

using BlackBIRDS
using DiffSIR
using ChainRulesCore
using DifferentiationInterface
using Distributions
using GraphNeuralNetworks
using Functors
using YAML
using Random

struct SIRJuneModel{T, G, I, S, P, L} <: BlackBIRDS.MultivariateStochasticModel{L}
    p::Vector{T}
    n::Int64
    graph::G
    venues::Vector{Symbol}
    infection_type::I
    discrete_sampler::S
    policies::P
    delta_t::Float64
    loss::L
end
Base.size(m::SIRJuneModel) = (2, m.n)
@functor SIRJuneModel (p,)

function generate_random_world_graph(
        n_agents, venues, fraction_population_per_venue, number_per_venue)
    n_venues = length(venues)
    num_nodes = Dict(:agent => n_agents)
    eindex_dict = Dict()
    for i in 1:n_venues
        n_agents_in_venue = Int(floor(n_agents * fraction_population_per_venue[i]))
        agents_in_venue = randperm(n_agents)[1:n_agents_in_venue]
        n_people_per_venue = number_per_venue[i]
        edges_venue = agents_in_venue, rand(1:n_people_per_venue, n_agents_in_venue)
        venue_symbol = Symbol(venues[i])
        eindex_dict[(:agent, :attends, venue_symbol)] = edges_venue
        eindex_dict[(venue_symbol, :attends, :agent)] = (edges_venue[2], edges_venue[1])
        num_nodes[venue_symbol] = n_people_per_venue
    end
    eindex = (k => v for (k, v) in eindex_dict)
    return GNNHeteroGraph(eindex; num_nodes)
end

function SIRJuneModel(
        graph, p, n_timesteps, discrete_sampler, loss; policies = DiffSIR.Policies(), delta_t = 1.0)
    venues = [k for k in keys(graph.num_nodes) if k != :agent]
    return SIRJuneModel(p, n_timesteps, graph, venues, DiffSIR.ConstantInfection(),
        discrete_sampler, policies, delta_t, loss)
end

function SIRJuneModel(config_path::String, loss)
    config = YAML.load_file(config_path)
    params = SIRParams(config)
    venues = collect(keys(params.betas_by_venue))
    betas = collect(values(params.betas_by_venue))
    p = [params.initial_infected, params.gamma, betas...]
    return SIRJuneModel(p, params.n_timesteps, params.graph, venues,
        params.infection_type, params.discrete_sampler, params.policies,
        params.delta_t, loss)
end

function Distributions.rand(sir::SIRJuneModel)
    initial_infected = sir.p[1]
    gamma = sir.p[2]
    betas_per_venue = sir.p[3:end]
    betas_by_venue = Dict(zip(sir.venues, betas_per_venue))
    results = run_sir(sir.graph, initial_infected, betas_by_venue, gamma, sir.delta_t,
        sir.n, sir.discrete_sampler, sir.infection_type, sir.policies)
    x = Matrix(hcat(results.delta_I_ts, results.delta_R_ts)')
    return x
    #return reshape(x, 1, 2, sir.n)
end

function ChainRulesCore.rrule(
        ::typeof(rand), d::SIRJuneModel{T, G, I, S, P, L}) where {
        T, G, I, S <: DiffSIR.SAD, P, L}
    v, grad = value_and_gradient(AutoStochasticAD(10), d, d.p)
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        d_tangent = Tangent{SIRJuneModel{T, G, I, S, P, L}}(;
            n = NoTangent(), p = grad' * y_tangent, graph = NoTangent(), venues = NoTangent(),
            infection_type = NoTangent(), discrete_sampler = NoTangent(),
            policies = NoTangent(), delta_t = NoTangent(), loss = NoTangent())
        return rand_tangent, d_tangent
    end
    return v, rand_pullback
end

function ChainRulesCore.rrule(
        ::typeof(rand), d::SIRJuneModel{T, G, I, S, P, L}) where {
        T, G, I, S, P, L}
    v, jacobians = BlackBIRDS.value_and_gradient(AutoForwardDiff(), d, d.p)
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        jacobian_1 = jacobians[1:d.n, :]
        jacobian_2 = jacobians[d.n+1:end, :]
        grad = jacobian_1' * y_tangent[1,:] + jacobian_2' * y_tangent[2,:]
        d_tangent = Tangent{SIRJuneModel{T, G, I, S, P, L}}(;
            n = NoTangent(), p = grad, graph = NoTangent(), venues = NoTangent(),
            infection_type = NoTangent(), discrete_sampler = NoTangent(),
            policies = NoTangent(), delta_t = NoTangent(), loss = NoTangent())
        return rand_tangent, d_tangent
    end
    return v, rand_pullback
end
#
end
