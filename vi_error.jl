using AdvancedVI
using ADTypes
using DynamicPPL
using DistributionsAD
using Distributions
using Bijectors
using Optimisers
using LinearAlgebra
using Zygote

function wrap_in_vec_reshape(f, in_size)
    vec_in_length = prod(in_size)
    reshape_inner = Bijectors.Reshape((vec_in_length,), in_size)
    out_size = Bijectors.output_size(f, in_size)
    vec_out_length = prod(out_size)
    reshape_outer = Bijectors.Reshape(out_size, (vec_out_length,))
    return reshape_outer ∘ f ∘ reshape_inner
end

function Bijectors.bijector(
        model::DynamicPPL.Model, ::Val{sym2ranges} = Val(false);
        varinfo = DynamicPPL.VarInfo(model)
) where {sym2ranges}
    num_params = sum([size(varinfo.metadata[sym].vals, 1) for sym in keys(varinfo.metadata)])

    dists = vcat([varinfo.metadata[sym].dists for sym in keys(varinfo.metadata)]...)

    num_ranges = sum([length(varinfo.metadata[sym].ranges)
                      for sym in keys(varinfo.metadata)])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1

    # ranges might be discontinuous => values are vectors of ranges rather than just ranges
    sym_lookup = Dict{Symbol, Vector{UnitRange{Int}}}()
    for sym in keys(varinfo.metadata)
        sym_lookup[sym] = Vector{UnitRange{Int}}()
        for r in varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            push!(sym_lookup[sym], ranges[range_idx])
            range_idx += 1
        end

        idx += varinfo.metadata[sym].ranges[end][end]
    end

    bs = map(tuple(dists...)) do d
        b = Bijectors.bijector(d)
        if d isa Distributions.UnivariateDistribution
            b
        else
            wrap_in_vec_reshape(b, size(d))
        end
    end

    if sym2ranges
        return (
            Bijectors.Stacked(bs, ranges),
            (; collect(zip(keys(sym_lookup), values(sym_lookup)))...)
        )
    else
        return Bijectors.Stacked(bs, ranges)
    end
end

function run_vi(;
	model,
	q,
	optimizer = Optimisers.Adam(1e-3),
	n_montecarlo,
	max_iter,
	adtype,
	gradient_method = "pathwise",
	entropy_estimation = AdvancedVI.ClosedFormEntropy(),
	transform = "auto",
)
	bijector_transf = inverse(bijector(model))
	q_transformed = transformed(q, bijector_transf)
	ℓπ = DynamicPPL.LogDensityFunction(model)
	elbo = AdvancedVI.ScoreGradELBO(n_montecarlo, entropy = entropy_estimation)
	q_untrained = deepcopy(q_transformed)

	q, _, stats, _ = AdvancedVI.optimize(
		ℓπ,
		elbo,
		q_transformed,
		max_iter;
		adtype,
		optimizer = optimizer,
	)
	return q, stats, q_untrained
end



##

function double_normal()
    return MvNormal([2.0, 3.0, 4.0], Diagonal(ones(3)))
end

@model function normal_model(data)
    p1 ~ filldist(Normal(0.0, 1.0), 2)
    p2 ~ filldist(Normal(0.0, 1.0), 1)
    ps = vcat(p1, p2)
    println(ps)
    for i in 1:size(data, 2)
        data[:, i] ~ MvNormal(ps, Diagonal(ones(3)))
    end
end

data = rand(double_normal(), 100)
model = normal_model(data)

##

d = 3
μ = zeros(d)
L = Diagonal(ones(d));
q = AdvancedVI.MeanFieldGaussian(μ, L)
optimizer = Optimisers.Adam(1e-3)

##
run_vi(
    model = model,
    q = q,
    optimizer = optimizer,
    n_montecarlo = 2,
    max_iter = 1,
    adtype = AutoZygote(),
    gradient_method = "score",
    entropy_estimation = AdvancedVI.ClosedFormEntropy(),
    transform = "auto",
)