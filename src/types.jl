export GradientMode, Pathwise, Score, AutoStochasticAD

abstract type GradientMode end
struct Pathwise <: GradientMode end
struct Score <: GradientMode end

struct AutoStochasticAD <: ADTypes.AbstractADType
    n_samples::Int64
end

