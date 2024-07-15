export NormalizingFlow, make_planar_flow

#struct NormalizingFlow{T, Q}
#    base_dist::T
#    transformations::Q
#end
#Functors.@functor NormalizingFlow (transformations,)

#function Distributions.rand(f::NormalizingFlow)
#    Distributions.rand(transformed(f.base_dist, f.transformations))
#end
#function Distributions.rand(f::NormalizingFlow, n::Int64)
#    Distributions.rand(transformed(f.base_dist, f.transformations), n)
#end
#Distributions.rand(rng::Random.AbstractRNG, f::NormalizingFlow) = rand(f)
#Distributions.rand(rng::Random.AbstractRNG, f::NormalizingFlow, n::Int64) = rand(f, n)
#function Distributions.logpdf(f::NormalizingFlow, x)
#    Distributions.logpdf(transformed(f.base_dist, f.transformations), x)
#end
#Base.length(f::NormalizingFlow) = length(f.base_dist)
#Base.size(f::NormalizingFlow) = size(f.base_dist)

function make_planar_flow(d, n = 5)
    b = PlanarLayer(d)
    for _ in 2:n
        b = b ∘ PlanarLayer(d)
    end
    base_dist = DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))
    return transformed(base_dist, b)
end
