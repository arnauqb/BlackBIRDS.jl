export GradientMode, Pathwise, Score

abstract type GradientMode end
struct Pathwise <: GradientMode end
struct Score <: GradientMode end
