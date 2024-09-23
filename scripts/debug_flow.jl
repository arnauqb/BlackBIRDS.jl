using BlackBIRDS
using Zygote
using Bijectors
using DynamicPPL


##
q = make_masked_affine_autoregressive_flow_torch(5, 4, 32);

##
@model function ppl_model(data)
    p = zeros(5)
    for i in 1:5
        p[i] ~ Beta(1.0, 1.0)
    end
    data ~ MvNormal(p, 1.0)
end
data = rand(MvNormal(0.5 .* ones(5), 1.0))

##
m = ppl_model(data)
b = bijector(m)
binv = inverse(b)

##
q_transformed = transformed(q, binv)

##
rand(q_transformed)

##
x = rand(q_transformed)
v, f = Zygote.pullback(logpdf, q_transformed, x)
f(v)