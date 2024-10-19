export create_planar_flow, AffineCoupling, create_radial_flow, create_neural_spline_flow
using NormalizingFlows

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
        input_dim::Int, hdims::Int, output_dim::Int; activation = Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim)
    )
end

struct NeuralSplineLayer{T, A <: Flux.Chain} <: Bijectors.Bijector
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
) where {T1 <: Int, T2 <: Real}
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
    hs = T[:, (K + 1):(2K)]
    ds = T[:, (2K + 1):(3K - 1)]
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

function create_neural_spline_flow(dim; n_layers = 2, hdims = 10, K = 8, B = 3)
    q0 = MvNormal(zeros(Float32, dim))
    Ls = [NeuralSplineLayer(dim, hdims, K, B, [1]) ∘
          NeuralSplineLayer(dim, hdims, K, B, [2]) for
          i in 1:n_layers]
    q0 = MvNormal(zeros(Float32, dim), I)
    flow = Bijectors.transformed(q0, ∘(Ls...))
    return flow
end

## AFFINE
"""
Affinecoupling layer 
"""
struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# let params track field s and t
Functors.@functor AffineCoupling (s, t)

function AffineCoupling(
        dim::Int,  # dimension of input
        hdims::Int, # dimension of hidden units for s and t
        mask_idx::AbstractVector # index of dimensione that one wants to apply transformations on
)
    cdims = length(mask_idx) # dimension of parts used to construct coupling law
    s = MLP_3layer(cdims, hdims, cdims)
    t = MLP_3layer(cdims, hdims, cdims)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = partition(af.mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return combine(af.mask, y₁, x₂, x₃)
end

function (af::AffineCoupling)(x::AbstractArray)
    return transform(af, x)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2))
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
        iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = Bijectors.partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2))
    return combine(af.mask, x_1, y_2, y_3), logjac
end

function Bijectors.logabsdetjac(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = partition(af.mask, x)
    logjac = sum(log ∘ abs, af.s(x_2))
    return logjac
end

function create_affine_coupling_flow(dim; nlayers=2)
    hdims = 20
    Ls = [AffineCoupling(dim, hdims, [1]) ∘ AffineCoupling(dim, hdims, [2]) for i in 1:nlayers]
    q0 = MvNormal(zeros(Float32, dim), I)
    flow = create_flow(Ls, q0)
    return flow
end
