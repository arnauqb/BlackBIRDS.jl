module BlackBIRDS

export StochasticModel, AutoStochasticAD, AutoForwardDiff, AutoZygote

using ADTypes
using AdvancedVI
using Bijectors
using ChainRulesCore
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
using Random
import Zygote

struct AutoStochasticAD <: ADTypes.AbstractADType
    n_samples::Int64
end

# for python models TODO: write as extension?
using PyCall
const torch = PyNULL()
const normflows = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
    copy!(normflows, pyimport("normflows"))
    @pyinclude(String(@__DIR__)*"/nf_wrapper.py")
end

include("models/core.jl")
include("utils.jl")
include("diff.jl")
include("losses.jl")
include("score.jl")
include("flows.jl")
include("flows_torch.jl")
include("vi.jl")

# models
include("models/RandomWalk.jl")
include("models/BrockHommes.jl")
include("models/SIRJune.jl")

end
