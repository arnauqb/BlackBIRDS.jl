using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra


##
struct LogTargetDensity{T}
    p::T
    n_timesteps::Int
end

function LogDensityProblems.logdensity(m::LogTargetDensity, y)
    loss = 0.0
    n_samples = 10
    for _ in 1:n_samples
        x = rand(rw) 
        loss += sum((x - y) .^ 2) / rw.n_timesteps^2
    end
    # score surrogate objective
    return -loss / n_samples
end
LogDensityProblems.dimension(p::LogTargetDensity) = p.n_timesteps
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

##
D = 1
initial_
