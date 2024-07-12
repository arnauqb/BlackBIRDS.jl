module BlackBIRDS

using Infiltrator

import ADTypes
import DifferentiationInterface
import ForwardDiff
using PyCall
import Optimisers
import StatsBase
import StochasticAD
import Zygote

# Write your package code here.
const torch = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
end

include("types.jl")
include("diff.jl")
include("vi.jl")

end
