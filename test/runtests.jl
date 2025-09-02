using BlackBIRDS
using Test

@testset "BlackBIRDS.jl" begin
    include("losses_test.jl")
    include("calibration_test.jl")
    # Write your tests here.
end
