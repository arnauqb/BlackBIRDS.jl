module BlackBIRDS

using Infiltrator

import ADTypes
using DifferentiationInterface
import Enzyme
import ForwardDiff
using PyCall
import Optimisers
import StatsBase

# Write your package code here.
const torch = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
end

include("types.jl")
include("vi.jl")

end
