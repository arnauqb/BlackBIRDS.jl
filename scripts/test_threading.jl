v, f = Zygote.pullback(BlackBIRDS.threaded_sampling, abm, 10)
f(v)