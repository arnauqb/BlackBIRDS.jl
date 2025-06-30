export create_planar_flow, AffineCoupling, create_radial_flow, create_neural_spline_flow, create_affine_coupling_flow
using NormalizingFlows
using Functors
using Bijectors: partition, combine, PartitionMask


function create_planar_flow(dim, n_layers::Int)
    q0 = MvNormal(zeros(Float32, dim))
    d = length(q0)
    Ls = [f32(PlanarLayer(d)) for _ in 1:n_layers]
    ts = fchain(Ls)
    return transformed(q0, ts)
end

function create_radial_flow(dim, n_layers::Int)
    q0 = MvNormal(zeros(Float32, dim))
    d = length(q0)
    Ls = [f32(RadialLayer(d)) for _ in 1:n_layers]
    ts = fchain(Ls)
    return transformed(q0, ts)
end

function MLP_3layer(
    input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim)
    )
end
function mlp3(input_dim::Int, hidden_dims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hidden_dims, activation),
        Flux.Dense(hidden_dims, hidden_dims, activation),
        Flux.Dense(hidden_dims, output_dim),
    )
end

struct NeuralSplineLayer{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int
    K::Int
    nn::AbstractVector{A} # networks that parmaterize the knots and derivatives
    B::T # bound of the knots
    mask::Bijectors.PartitionMask
end

function NeuralSplineLayer(
    dim::T1,  # dimension of input
    hdims::T1, # dimension of hidden units for s and t
    K::T1, # number of knots
    B::T2, # bound of the knots
    mask_idx::AbstractVector{<:Int} # index of dimensione that one wants to apply transformations on
) where {T1<:Int,T2<:Real}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    nn = fill(MLP_3layer(input_dims, hdims, 3K - 1), num_of_transformed_dims)
    mask = Bijectors.PartitionMask(dim, mask_idx)
    return NeuralSplineLayer(dim, K, nn, B, mask)
end

Functors.@functor NeuralSplineLayer (nn,)

# define forward and inverse transformation
function instantiate_rqs(nsl::NeuralSplineLayer, x::AbstractVector)
    # instantiate rqs knots and derivatives
    T = permutedims(reduce(hcat, map(nn -> nn(x), nsl.nn)))
    K, B = nsl.K, nsl.B
    ws = T[:, 1:K]
    hs = T[:, (K+1):(2K)]
    ds = T[:, (2K+1):(3K-1)]
    return Bijectors.RationalQuadraticSpline(ws, hs, ds, B)
end

function Bijectors.transform(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    # instantiate rqs knots and derivatives
    rqs = instantiate_rqs(nsl, x_2)
    y_1 = Bijectors.transform(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3)
end

function Bijectors.transform(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = Bijectors.partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    x1 = Bijectors.transform(Inverse(rqs), y1)
    return Bijectors.combine(nsl.mask, x1, y2, y3)
end

function (nsl::NeuralSplineLayer)(x::AbstractVector)
    return Bijectors.transform(nsl, x)
end

# define logabsdetjac
function Bijectors.logabsdetjac(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    logjac = logabsdetjac(Inverse(rqs), y1)
    return logjac
end

function Bijectors.with_logabsdet_jacobian(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    y_1, logjac = with_logabsdet_jacobian(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3), logjac
end

function create_neural_spline_flow(dim; n_layers=2, hdims=10, K=8, B=3)
    q0 = MvNormal(zeros(Float32, dim))
    Ls = [NeuralSplineLayer(dim, hdims, K, B, [1]) ∘
          NeuralSplineLayer(dim, hdims, K, B, [2]) for
          i in 1:n_layers]
    q0 = MvNormal(zeros(Float32, dim), I)
    flow = Bijectors.transformed(q0, ∘(Ls...))
    return flow
end

## Real NVP
##################################
# define affine coupling layer using Bijectors.jl interface
#################################
struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# let params track field s and t
@functor AffineCoupling (s, t)

function AffineCoupling(
    dim::Int,  # dimension of input
    hdims::Int, # dimension of hidden units for s and t
    mask_idx::AbstractVector, # index of dimensione that one wants to apply transformations on
)
    cdims = length(mask_idx) # dimension of parts used to construct coupling law
    input_dims = dim - cdims # dimension of the input to the neural networks
    s = mlp3(input_dims, hdims, cdims)
    t = mlp3(input_dims, hdims, cdims)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

function (af::AffineCoupling)(x::AbstractArray)
    return transform(af, x)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2)) # this is a scalar
    return Bijectors.combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractMatrix)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2); dims=1) # 1 × size(x, 2)
    return Bijectors.combine(af.mask, y_1, x_2, x_3), vec(logjac)
end


function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2))
    return Bijectors.combine(af.mask, x_1, y_2, y_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractMatrix
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2); dims=1)
    return Bijectors.combine(af.mask, x_1, y_2, y_3), vec(logjac)
end

function create_flow(Ls, q₀)
    ts =  reduce(∘, Ls)
    return transformed(q₀, ts)
end

function create_affine_coupling_flow(dim; nlayers=2)
    hdims = 20
    # Create a sequence of coupling layers that transform each dimension in sequence
    Ls = []
    for i in 1:nlayers
        # For each layer, create a sequence of coupling layers that transform each dimension
        layer = reduce(∘, [AffineCoupling(dim, hdims, [j]) for j in 1:dim])
        push!(Ls, layer)
    end
    q0 = MvNormal(zeros(Float32, dim), I)
    flow = create_flow(Ls, q0)
    return flow
end
