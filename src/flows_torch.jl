export PyTorchFlow, make_real_nvp_flow_torch, make_masked_affine_autoregressive_flow_torch,
    make_neural_spline_flow_torch, make_planar_flow_torch
export serialize_flow, deserialize_flow

# wrapper around normflows
import ForwardDiff: Dual, Partials

struct PyTorchFlow{T,N,M} <: Distributions.ContinuousMultivariateDistribution
    func_sampler::PyObject
    func_logpdf::PyObject
    buffers::NTuple{N,PyObject}
    indices::Vector{Tuple{Int64,Int64}}
    params::NTuple{M,PyObject}
    params_flat::Vector{T}
    output_length::Int64
end
Distributions.length(d::PyTorchFlow) = d.output_length
Functors.@functor PyTorchFlow (params_flat,)

function PyTorchFlow(flow::PyObject)
    func_sampler, func_logpdf, params, buffers = py"deconstruct_flow"(flow)
    params_flat, indices = py"flatten_params"(params)
    params_flat = params_flat.numpy()
    return PyTorchFlow(
        func_sampler, func_logpdf, buffers, indices, params, params_flat, py"int"(flow.q0.d))
end

function Distributions.rand(dist::PyTorchFlow)
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, 1).flatten().numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow) = rand(dist)

function Distributions.rand(dist::PyTorchFlow, n::Int)
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat, dtype=torch.float), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, n).numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow, n::Int) = rand(dist, n)

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractArray{<:Real})
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat, dtype=torch.float), dist.indices, dist.params)
    return dist.func_logpdf(
        params, dist.buffers, torch.tensor(x, dtype=torch.float)).numpy()
end

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractVector{<:Real})
    logpdf(dist, reshape(x, length(x), 1))[1]
end

## Zygote rules
function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow, n::Int64)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat, dtype=torch.float),
        d.buffers, d.indices, n)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent, dtype=torch.float))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad.numpy())
        return NoTangent(), d_tangent, NoTangent()
    end
    return samples.numpy(), rand_pullback
end

function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat, dtype=torch.float),
        d.buffers, d.indices, 1)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent, dtype=torch.float).reshape(-1, 1))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad.numpy())
        return NoTangent(), d_tangent
    end
    return samples.numpy()[:], rand_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractVector{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat, dtype=torch.float), d.buffers,
        d.indices, torch.tensor(x, dtype=torch.float).reshape(-1, 1))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent, dtype=torch.float).reshape(1))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, x_tangent
    end
    return lps.numpy()[1], logpdf_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractMatrix{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat, dtype=torch.float),
        d.buffers, d.indices, torch.tensor(x))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, NoTangent(), x_tangent
    end
    return lps.numpy(), logpdf_pullback
end

## Flow constructors

function make_real_nvp_flow_torch(dim, n_layers, hidden_dim)
    flows = []
    for i in 1:n_layers
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = normflows.nets.MLP(
            [Int(dim / 2), hidden_dim, hidden_dim, dim], init_zeros=true)
        # Add flow layer
        push!(flows, normflows.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        push!(flows, normflows.flows.Permute(dim, mode="swap"))
    end
    base = normflows.distributions.base.DiagGaussian(dim)
    # Construct flow model
    flow_py = normflows.NormalizingFlow(base, flows)
    for param in collect(flow_py.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(flow_py)
end

function _make_masked_affine_autoregressive_flow_torch(; dim, n_layers, n_units)
    flows = []
    for i in 1:n_layers
        push!(flows,
            normflows.flows.MaskedAffineAutoregressive(dim, n_units, num_blocks=2))
        push!(flows, normflows.flows.LULinearPermute(dim))
    end
    q0 = normflows.distributions.DiagGaussian(dim)
    nfm = normflows.NormalizingFlow(q0=q0, flows=flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    return nfm
end

function make_masked_affine_autoregressive_flow_torch(; dim, n_layers, n_units)
    nfm = _make_masked_affine_autoregressive_flow_torch(;dim, n_layers, n_units)
    # remove the gradient from the parameters
    flow = PyTorchFlow(nfm)
    return flow
end

function make_neural_spline_flow_torch(dim, n_layers, hidden_units, hidden_layers=2)
    # Define flows
    K = n_layers

    latent_size = dim
    flows = []
    for i in 1:K
        push!(flows,
            normflows.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units))
        push!(flows, normflows.flows.LULinearPermute(latent_size))
    end
    # Set base distribuiton
    q0 = normflows.distributions.DiagGaussian(dim, trainable=false)
    # Construct flow model
    nfm = normflows.NormalizingFlow(q0=q0, flows=flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end

    return PyTorchFlow(nfm)
end

function make_planar_flow_torch(dim, n_layers)
    flows = []
    for i in 1:n_layers
        push!(flows, normflows.flows.Planar((dim,), act="leaky_relu"))
    end
    q0 = normflows.distributions.DiagGaussian(dim, trainable=false)
    nfm = normflows.NormalizingFlow(q0=q0, flows=flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(nfm)
end

## Serialization


function serialize_flow(
    flow::MultivariateTransformed{<:PyTorchFlow}, hyper_parameters, function_call)
    transform = flow.transform
    flow_parameters = flow.dist.params_flat
    buffers_arrays = [b.numpy() for b in flow.dist.buffers]
    params_arrays = [p.numpy() for p in flow.dist.params]
    dict = Dict(:transform => transform, :params_flat => flow.dist.params_flat,
        :hyper_parameters => hyper_parameters, :function_call => function_call,
        :buffers => buffers_arrays, :params => params_arrays, :indices => flow.dist.indices)
    return dict
end

function save_flow(serialized_dict, path)
    serialize(path, serialized_dict)
end

function deserialize_flow(flow_serialized::Dict)
    transform = flow_serialized[:transform]
    params = flow_serialized[:params]
    params = tuple([torch.tensor(p, dtype=torch.float) for p in params]...)
    params_flat = flow_serialized[:params_flat]
    hyper_parameters = flow_serialized[:hyper_parameters]
    function_call = flow_serialized[:function_call]
    buffers = flow_serialized[:buffers]
    buffers = tuple([torch.tensor(b, dtype=torch.float) for b in buffers]...)
    indices = flow_serialized[:indices]
    flow = _make_masked_affine_autoregressive_flow_torch(; hyper_parameters...)
    func_sampler, func_logpdf, _, _ = py"deconstruct_flow"(flow)
    full_flow = PyTorchFlow(
        func_sampler, func_logpdf, buffers, indices, params, params_flat, py"int"(flow.q0.d))
    return transformed(full_flow, transform)
end

function deserialize_flow(path::String)
    flow_serialized = deserialize(path)
    return deserialize_path(flow_serialized)
end