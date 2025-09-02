module BlackBIRDS

export StochasticModel, AutoStochasticAD, AutoForwardDiff, AutoZygote

using ADTypes
using AdvancedVI
using AdvancedHMC
using Bijectors
using ChainRulesCore
using DiffABM 
using Flux
import DifferentiationInterface
using DiffResults
using Distributions
import DistributionsAD
import DynamicPPL
import Functors
import ForwardDiff
import LogDensityProblems
using LinearAlgebra
import Optimisers
using SimpleUnPack
using ProgressMeter
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

abstract type StochasticModel{B, L} <: Distributions.ContinuousMatrixDistribution end

include("utils.jl")
include("diff.jl")
include("losses.jl")
include("flows.jl")
include("flows_torch.jl")
include("vi.jl")
include("hmc.jl")
include("abms.jl")

end
