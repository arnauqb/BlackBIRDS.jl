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

abstract type StochasticModel{B, L} <: Distributions.ContinuousMatrixDistribution end

include("utils.jl")
include("diff.jl")
include("losses.jl")
include("flows.jl")
include("vi.jl")
include("hmc.jl")
include("abms.jl")

end
