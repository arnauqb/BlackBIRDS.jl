export PyTorchFlow, make_real_nvp_flow_torch, make_masked_affine_autoregressive_flow_torch

# wrapper around normflows

struct PyTorchFlow{T} <: Distributions.ContinuousMultivariateDistribution
    flow::PyObject
    flat_params::Vector{T}
end

function PyTorchFlow(flow::PyObject)
    flat_params = torch.nn.utils.parameters_to_vector(flow.parameters()).numpy()
    return PyTorchFlow(flow, flat_params)
end

function make_reconstruct_f(flow)
    function reconstruct_f(parameters)
        #py"assign_params_to_flow"(flow.flow, torch.tensor(parameters))
        return PyTorchFlow(flow.flow, parameters)
    end
    return reconstruct_f
end

function Optimisers.destructure(flow::PyTorchFlow)
    return flow.flat_params, make_reconstruct_f(flow)
end

function Distributions.rand(dist::PyTorchFlow)
    @pywith torch.no_grad() begin
        py"assign_params_to_flow"(dist.flow, torch.tensor(dist.flat_params))
        return dist.flow.sample(1)[1].numpy()[:]
    end
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow) = rand(dist)

function Distributions.rand(dist::PyTorchFlow, n::Int)
    @pywith torch.no_grad() begin
        py"assign_params_to_flow"(dist.flow, torch.tensor(dist.flat_params))
        return dist.flow.sample(n)[1].t().numpy()
    end
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow, n::Int) = rand(dist, n)

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractArray{<:Real})
    @pywith torch.no_grad() begin
        py"assign_params_to_flow"(dist.flow, torch.tensor(dist.flat_params))
        return dist.flow.log_prob(torch.tensor(x).t()).numpy()
    end
end

Distributions.logpdf(dist::PyTorchFlow, x::AbstractVector{<:Real}) = logpdf(dist, reshape(x, length(x), 1))

function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow, n::Int64)
    samples, torch_pullback = py"sample_pullback"(d.flow, torch.tensor(d.flat_params), n)
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        grad = torch_pullback(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(; flow = NoTangent(), flat_params = grad)
        n_tangent = NoTangent()
        return rand_tangent, d_tangent, n_tangent
    end
    return samples, rand_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractMatrix{T}) where {T}
    lp, torch_pullback = py"logpdf_pullback"(d.flow, torch.tensor(d.flat_params), torch.tensor(x).t())
    function logpdf_pullback(y_tangent)
        logpdf_tangent = NoTangent()
        grad = torch_pullback(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(; flow = NoTangent(), flat_params = grad)
        x_tangent = NoTangent()
        return logpdf_tangent, d_tangent, x_tangent
    end
    return lp, logpdf_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractVector{T}) where {T}
    lp, torch_pullback = py"logpdf_pullback"(d.flow, torch.tensor(d.flat_params), torch.tensor(x).reshape(1, -1))
    lp = lp[1]
    function logpdf_pullback(y_tangent)
        logpdf_tangent = NoTangent()
        grad = torch_pullback(torch.tensor(y_tangent).reshape(1))
        d_tangent = Tangent{PyTorchFlow}(; flow = NoTangent(), flat_params = grad)
        x_tangent = NoTangent()
        return logpdf_tangent, d_tangent, x_tangent
    end
    return lp, logpdf_pullback
end

function make_real_nvp_flow_torch(dim, n_layers, hidden_dim)
    base = normflows.distributions.base.DiagGaussian(dim)
    flows = []
    for _ in 1:n_layers
        param_map = normflows.nets.MLP([2, hidden_dim, hidden_dim, dim], init_zeros = true)
        push!(flows, normflows.flows.AffineCouplingBlock(param_map))
        push!(flows, normflows.flows.Permute(dim, mode = "swap"))
    end
    flow_py = normflows.NormalizingFlow(base, flows)
    return PyTorchFlow(flow_py)
end

function make_masked_affine_autoregressive_flow_torch(dim, n_layers, hidden_dim)
    flows = []
    for i in 1:n_layers
        push!(flows,
            normflows.flows.MaskedAffineAutoregressive(dim, hidden_dim, num_blocks = 2))
        push!(flows, normflows.flows.LULinearPermute(dim))
    end
    #flows += [Sigmoid(min_values=-3.0 * torch.ones(6), max_values=torch.zeros(6))]
    q0 = normflows.distributions.DiagGaussian(dim)
    nfm = normflows.NormalizingFlow(q0 = q0, flows = flows)
    # remove the gradient from the parameters
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(nfm)
end