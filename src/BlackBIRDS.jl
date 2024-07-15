module BlackBIRDS

export StochasticModel, AutoStochasticAD, AutoForwardDiff, AutoZygote

using ADTypes
using AdvancedVI
using Bijectors
import ChainRulesCore
using Flux
import DifferentiationInterface
using DiffResults
using Distributions
import DistributionsAD
import DynamicPPL
import Functors
import ForwardDiff
import LogDensityProblems
import Optimisers
using SimpleUnPack
import StochasticAD
import Random
import Zygote

abstract type StochasticModel{L} <: Distributions.ContinuousMultivariateDistribution end

struct AutoStochasticAD <: ADTypes.AbstractADType
    n_samples::Int64
end


# for python models TODO: write as extension?
using PyCall
const torch = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
end

include("utils.jl")
include("diff.jl")
include("losses.jl")
include("score.jl")
include("flows.jl")
include("vi.jl")

# models
include("models/RandomWalk.jl")

end
