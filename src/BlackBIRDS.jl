module BlackBIRDS

export AutoForwardDiff, AutoZygote, AutoEnzyme

using Infiltrator

import ADTypes
import DifferentiationInterface
using DifferentiationInterface: AutoForwardDiff, AutoZygote, AutoEnzyme
import ForwardDiff
using PyCall
import Optimisers
import StatsBase
import StochasticAD
import Zygote
import Distributions
using DistributionsAD

# Write your package code here.
const torch = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
end

include("types.jl")
include("diff.jl")
include("dist.jl")
include("vi.jl")

end
