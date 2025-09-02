export drop_gradient

drop_gradient(x) = ChainRulesCore.@ignore_derivatives x
drop_gradient(x::ForwardDiff.Dual) = ForwardDiff.value(x)
drop_gradient(x::StochasticAD.StochasticTriple) = StochasticAD.value(x)