module BlackBIRDS

export StochasticModel, AutoStochasticAD, AutoForwardDiff, AutoZygote

using ADTypes
using AdvancedVI
import Bijectors
import ChainRulesCore
using Flux
import DifferentiationInterface
using DiffResults
import Distributions
import DistributionsAD
import DynamicPPL
import ForwardDiff
using PyCall
import Optimisers
using SimpleUnPack
import StochasticAD
import Random
import Zygote

abstract type StochasticModel{L} <: Distributions.ContinuousMultivariateDistribution end

struct AutoStochasticAD <: ADTypes.AbstractADType
    n_samples::Int64
end


# Write your package code here.
const torch = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
end

include("utils.jl")
include("diff.jl")
include("losses.jl")
include("score.jl")
include("vi.jl")

# models
include("models/RandomWalk.jl")

end
