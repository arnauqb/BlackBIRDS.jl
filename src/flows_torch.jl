export PyTorchFlow, make_real_nvp_flow_torch, make_masked_affine_autoregressive_flow_torch,
       make_neural_spline_flow_torch, make_planar_flow_torch

# wrapper around normflows
import ForwardDiff: Dual, Partials

struct PyTorchFlow{T, N, M} <: Distributions.ContinuousMultivariateDistribution
    func_sampler::PyObject
    func_logpdf::PyObject
    buffers::NTuple{N, PyObject}
    indices::Vector{Tuple{Int64, Int64}}
    params::NTuple{M, PyObject}
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
        torch.tensor(dist.params_flat, dtype = torch.float), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, n).numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow, n::Int) = rand(dist, n)

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractArray{<:Real})
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat, dtype = torch.float), dist.indices, dist.params)
    return dist.func_logpdf(
        params, dist.buffers, torch.tensor(x, dtype = torch.float)).numpy()
end

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractVector{<:Real})
    logpdf(dist, reshape(x, length(x), 1))[1]
end

## Zygote rules
function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow, n::Int64)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat, dtype = torch.float),
        d.buffers, d.indices, n)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent, dtype = torch.float))
        d_tangent = Tangent{PyTorchFlow}(; params_flat = grad.numpy())
        return NoTangent(), d_tangent, NoTangent()
    end
    return samples.numpy(), rand_pullback
end

function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat, dtype = torch.float),
        d.buffers, d.indices, 1)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent, dtype = torch.float).reshape(-1, 1))
        d_tangent = Tangent{PyTorchFlow}(; params_flat = grad.numpy())
        return NoTangent(), d_tangent
    end
    return samples.numpy()[:], rand_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractVector{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat, dtype = torch.float), d.buffers,
        d.indices, torch.tensor(x, dtype = torch.float).reshape(-1, 1))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent, dtype = torch.float).reshape(1))
        d_tangent = Tangent{PyTorchFlow}(; params_flat = grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, x_tangent
    end
    return lps.numpy()[1], logpdf_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractMatrix{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat, dtype = torch.float),
        d.buffers, d.indices, torch.tensor(x))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(; params_flat = grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, NoTangent(), x_tangent
    end
    return lps.numpy(), logpdf_pullback
end

## Flow constructors

function make_real_nvp_flow_torch(dim, n_layers, hidden_dim)
    base = normflows.distributions.base.DiagGaussian(dim)
    flows = []
    for _ in 1:n_layers
        param_map = normflows.nets.MLP([2, hidden_dim, hidden_dim, dim], init_zeros = true)
        push!(flows, normflows.flows.AffineCouplingBlock(param_map))
        push!(flows, normflows.flows.Permute(dim, mode = "swap"))
    end
    flow_py = normflows.NormalizingFlow(base, flows)
    # remove the gradient from the parameters
    for param in collect(flow_py.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(flow_py)
end

function make_masked_affine_autoregressive_flow_torch(
        dim, n_layers, hidden_dim)
    flows = []
    for i in 1:n_layers
        push!(flows,
            normflows.flows.MaskedAffineAutoregressive(dim, hidden_dim, num_blocks = 2))
        push!(flows, normflows.flows.LULinearPermute(dim))
    end
    q0 = normflows.distributions.DiagGaussian(dim)
    nfm = normflows.NormalizingFlow(q0 = q0, flows = flows)
    # remove the gradient from the parameters
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    flow = PyTorchFlow(nfm)
    return flow
end

function make_neural_spline_flow_torch(dim, n_layers, hidden_units, hidden_layers = 2)
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
    q0 = normflows.distributions.DiagGaussian(dim, trainable = false)
    # Construct flow model
    nfm = normflows.NormalizingFlow(q0 = q0, flows = flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end

    return PyTorchFlow(nfm)
end

function make_planar_flow_torch(dim, n_layers)
    flows = []
    for i in 1:n_layers
        push!(flows, normflows.flows.Planar((dim,), act = "leaky_relu"))
    end
    q0 = normflows.distributions.DiagGaussian(dim, trainable = false)
    nfm = normflows.NormalizingFlow(q0 = q0, flows = flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(nfm)
end
