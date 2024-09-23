using BlackBIRDS
using DiffABM
using Flux
using Functors
using Zygote

##
params = GameOfLifeParams(100, 10, rand(10), [0.1], GS(0.1))
abm = ABM(params, AutoForwardDiff(), MSELoss(10))
Flux.trainable(abm)


##
Zygote.refresh()
v, f = Zygote.pullback(rand, abm)
f(v)

