export PyTorchFlow, make_real_nvp_flow_torch, make_masked_affine_autoregressive_flow_torch

# wrapper around normflows

struct PyTorchFlow{T, N, M} <: Distributions.ContinuousMultivariateDistribution
    func_sampler::PyObject
    func_logpdf::PyObject
    buffers::NTuple{N, PyObject}
    indices::Vector{Tuple{Int64, Int64}}
    params::NTuple{M, PyObject}
    params_flat::Vector{T}
end
Functors.@functor PyTorchFlow (params_flat,)

function PyTorchFlow(flow::PyObject)
    func_sampler, func_logpdf, params, buffers = py"deconstruct_flow"(flow)
    params_flat, indices = py"flatten_params"(params)
    params_flat = params_flat.numpy()
    return PyTorchFlow(func_sampler, func_logpdf, buffers, indices, params, params_flat)
end

function Distributions.rand(dist::PyTorchFlow)
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, 1).flatten().numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow) = rand(dist)

function Distributions.rand(dist::PyTorchFlow, n::Int)
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, n).numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow, n::Int) = rand(dist, n)

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractArray{<:Real})
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat), dist.indices, dist.params)
    return dist.func_logpdf(params, dist.buffers, torch.tensor(x)).numpy()
end

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractVector{<:Real})
    logpdf(dist, reshape(x, length(x), 1))[1]
end

function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow, n::Int64)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat), d.buffers, d.indices, n)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(;
            func_sampler = NoTangent(), func_logpdf = NoTangent(), buffers = NoTangent(),
            indices = NoTangent(), params = NoTangent(), params_flat = grad.numpy())
        return NoTangent(), d_tangent, NoTangent()
    end
    return samples.numpy(), rand_pullback
end

function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat), d.buffers, d.indices, 1)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent).reshape(-1, 1))
        d_tangent = Tangent{PyTorchFlow}(;
            func_sampler = NoTangent(), func_logpdf = NoTangent(), buffers = NoTangent(),
            indices = NoTangent(), params = NoTangent(), params_flat = grad.numpy())
        return NoTangent(), d_tangent
    end
    return samples.numpy()[:], rand_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractVector{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat), d.buffers,
        d.indices, torch.tensor(x).reshape(-1, 1))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent).reshape(1))
        d_tangent = Tangent{PyTorchFlow}(;
            func_sampler = NoTangent(), func_logpdf = NoTangent(), buffers = NoTangent(),
            indices = NoTangent(), params = NoTangent(), params_flat = grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, x_tangent
    end
    return lps.numpy()[1], logpdf_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractMatrix{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat), d.buffers, d.indices, torch.tensor(x))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(;
            func_sampler = NoTangent(), func_logpdf = NoTangent(), buffers = NoTangent(),
            indices = NoTangent(), params = NoTangent(), params_flat = grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, NoTangent(), x_tangent
    end
    return lps.numpy(), logpdf_pullback
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

function make_masked_affine_autoregressive_flow_torch(
        dim, n_layers, hidden_dim; param_ranges = nothing)
    flows = []
    for i in 1:n_layers
        push!(flows,
            normflows.flows.MaskedAffineAutoregressive(dim, hidden_dim, num_blocks = 2))
        push!(flows, normflows.flows.LULinearPermute(dim))
    end
    if param_ranges !== nothing
        push!(flows,
            py"Sigmoid"(min_values = torch.tensor(param_ranges[1], dtype=torch.float),
                max_values = torch.tensor(param_ranges[2], dtype = torch.float)))
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