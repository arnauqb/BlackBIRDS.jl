export make_planar_flow, AffineCoupling, make_affine_flow

function MLP_3layer(
        input_dim::Int, hdims::Int, output_dim::Int; activation = Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim)
    )
end

function make_planar_flow(d, n = 5)
    b = PlanarLayer(d)
    for _ in 2:n
        b = b ∘ PlanarLayer(d)
    end
    base_dist = DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))
    return transformed(base_dist, b)
end

struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# to apply functions to the parameters that are contained in AffineCoupling.s and AffineCoupling.t,
# and to re-build the struct from the parameters, we use the functor interface of `Functors.jl`
# see https://fluxml.ai/Flux.jl/stable/models/functors/#Functors.functor
Functors.@functor AffineCoupling (s, t)

function AffineCoupling(
        dim::Int,  # dimension of input
        hdims::Int, # dimension of hidden units for s and t
        mask_idx::AbstractVector # index of dimension that one wants to apply transformations on
)
    cdims = length(mask_idx) # dimension of parts used to construct coupling law
    s = MLP_3layer(cdims, hdims, cdims)
    t = MLP_3layer(cdims, hdims, cdims)
    mask = Bijectors.PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = Bijectors.partition(af.mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return Bijectors.combine(af.mask, y₁, x₂, x₃)
end

function Bijectors.transform(iaf::Inverse{<:AffineCoupling}, y::AbstractVector)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = Bijectors.partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    return Bijectors.combine(af.mask, x_1, y_2, y_3)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2))
    return Bijectors.combine(af.mask, y_1, x_2, x_3), logjac
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
    return Bijectors.combine(af.mask, x_1, y_2, y_3), logjac
end

function make_affine_flow(dim, nlayers, hdims)
    Ls = []
    for _ in 1:nlayers
        idx = randperm(dim)[1:div(dim, 2)]
        push!(Ls, AffineCoupling(dim, hdims, idx))
    end
    ts = reduce(∘, Ls)
    base_dist = DistributionsAD.TuringDiagMvNormal(zeros(Float32, dim), ones(Float32, dim))
    return transformed(base_dist, ts)
end
