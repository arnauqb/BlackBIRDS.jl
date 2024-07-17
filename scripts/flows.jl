using BlackBIRDS
using Distributions
using Zygote
using PyCall

##
q = make_planar_flow(3, 8);
flow = make_masked_affine_autoregressive_flow(3, 2, 4);

##
