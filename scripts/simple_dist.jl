using ChainRulesCore
using Distributions
using Random
using Zygote

struct A{T}
    p::Vector{T}
end
function f(A)
    return 2 * A.p[1]
end
function ChainRulesCore.rrule(::typeof(f), a::A{T}) where {T}
    function rand_pullback(y_tangent)
        rand_tangent = NoTangent()
        d_tangent = Tangent{A{T}}(; p = 2 * y_tangent)
        return rand_tangent, d_tangent
    end
    v = f(a)
    return v, rand_pullback
end
a = A([0.1])
Zygote.gradient(x -> f(x)[1], a) # returns ((p = [2.0],),)
#Zygote.jacobian(f, a) # returns nothing

y, back = Zygote.pullback(f, a)

