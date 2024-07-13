##
using PyCall
nf = pyimport("normflows")

##
function make_flow(n_parameters)
    K = 8
    flows = []
    for i in 1:K
        push!(flows, nf.flows.MaskedAffineAutoregressive(n_parameters, 20, num_blocks=2))
        push!(flows, nf.flows.Permute(n_parameters, mode="swap"))
    end
    q0 = nf.distributions.DiagGaussian(n_parameters)
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm
end

##
flow = make_flow(2)

##
flow.sample(100)[1].sum().backward()
asd

for p in flow.parameters()
    println(p.grad)
end
