export value_and_gradient
using StochasticAD: StochasticTriple, value, delta, combine

function value_and_gradient(ad::ADTypes.AbstractADType, func, params)
    return DifferentiationInterface.value_and_jacobian(func, ad, params)
end

function value_and_gradient(ad::AutoStochasticAD, func, params)
    n_samples = ad.n_samples
    st_samples = Matrix{Float64}[]
    st_samples = fetch.([Threads.@spawn hcat(StochasticAD.derivative_estimate(
        func, params)...) for _ in 1:n_samples])
    value = func(params)
    jacobian = sum(st_samples) / n_samples
    return value, jacobian
end

"""
rules for multiindexing with StochasticTriples
"""
function Base.getindex(
        C::AbstractArray, st::StochasticTriple{T, V, FIs}, i2) where {T, V, FIs}
    val = C[st.value, i2]
    do_map = (Δ, state) -> begin
        return value(C[st.value + Δ, i2], state) - value(val, state)
    end
    is_assigned(C, i) = 1 <= i <= size(C, 1)
    # TODO: below doesn't support sparse arrays, use something like nextind
    deriv = δ -> begin
        scale = if is_assigned(C, st.value + 1) && is_assigned(C, st.value - 1)
            1 / 2 * (value(C[st.value + 1, i2]) - value(C[st.value - 1, i2]))
        elseif is_assigned(C, st.value + 1)
            value(C[st.value + 1, i2]) - value(C[st.value, i2])
        elseif is_assigned(C, st.value - 1)
            value(C[st.value, i2]) - value(C[st.value - 1, i2])
        else
            zero(eltype(C))
        end
        return scale * δ
    end

    Δs = StochasticAD.map_Δs(do_map, st.Δs; deriv, out_rep = value(val))
    if val isa StochasticTriple
        Δs = combine((Δs, val.Δs))
    end
    return StochasticTriple{T}(value(val), delta(val), Δs)
end

function Base.getindex(
        C::AbstractArray, i1, st::StochasticTriple{T, V, FIs}) where {T, V, FIs}
    val = C[i1, st.value]
    do_map = (Δ, state) -> begin
        return value(C[i1, st.value + Δ], state) - value(val, state)
    end
    is_assigned(C, i) = 1 <= i <= size(C, 2)
    # TODO: below doesn't support sparse arrays, use something like nextind
    deriv = δ -> begin
        scale = if is_assigned(C, st.value + 1) && is_assigned(C, st.value - 1)
            1 / 2 * (value(C[i1, st.value + 1]) - value(C[i1, st.value - 1]))
        elseif is_assigned(C, st.value + 1)
            value(C[i1, st.value + 1]) - value(C[i1, st.value])
        elseif is_assigned(C, st.value - 1)
            value(C[i1, st.value]) - value(C[i1, st.value - 1])
        else
            zero(eltype(C))
        end
        return scale * δ
    end

    Δs = StochasticAD.map_Δs(do_map, st.Δs; deriv, out_rep = value(val))
    if val isa StochasticTriple
        Δs = combine((Δs, val.Δs))
    end
    return StochasticTriple{T}(value(val), delta(val), Δs)
end

function Base.getindex(C::AbstractArray, i1::StochasticTriple{T, V, FIs},
        i2::StochasticTriple{T, V, FIs}) where {T, V, FIs}
    val = C[i1.value, i2.value]
    do_map_1 = (Δ, state) -> begin
        return value(C[i1.value + Δ, i2.value], state) - value(val, state)
    end
    do_map_2 = (Δ, state) -> begin
        return value(C[i1.value, i2.value + Δ], state) - value(val, state)
    end
    is_assigned_1(C, i) = 1 <= i <= size(C, 1)
    is_assigned_2(C, i) = 1 <= i <= size(C, 2)
    # TODO: below doesn't support sparse arrays, use something like nextind
    deriv_1 = δ -> begin
        scale = if is_assigned_1(C, i1.value + 1) && is_assigned_1(C, i1.value - 1)
            1 / 2 * (value(C[i1.value + 1, i2.value]) - value(C[i1.value - 1, i2.value]))
        elseif is_assigned_1(C, i1.value + 1)
            value(C[i1.value + 1, i2.value]) - value(C[i1.value, i2.value])
        elseif is_assigned_1(C, st.value - 1)
            value(C[i1.value, i2.value]) - value(C[i1.value - 1, i2.value])
        else
            zero(eltype(C))
        end
        return scale * δ
    end
    deriv_2 = δ -> begin
        scale = if is_assigned_2(C, i2.value + 1) && is_assigned_2(C, i2.value - 1)
            1 / 2 * (value(C[i1.value, i2.value + 1]) - value(C[i1.value, i2.value - 1]))
        elseif is_assigned_2(C, i2.value + 1)
            value(C[i1.value, i2.value + 1]) - value(C[i1.value, i2.value])
        elseif is_assigned_2(C, i2.value - 1)
            value(C[i1.value, i2.value]) - value(C[i1.value, i2.value - 1])
        else
            zero(eltype(C))
        end
        return scale * δ
    end

    Δs_1 = StochasticAD.map_Δs(do_map_1, i1.Δs; deriv_1, out_rep = value(val))
    Δs_2 = StochasticAD.map_Δs(do_map_2, i2.Δs; deriv_2, out_rep = value(val))
    # randomly choose one of the two to be the "main" perturbation. Should be weighted by their relative weight
    w1 = Δs_1.state.weight
    w2 = Δs_2.state.weight
    if rand() < w1 / (w1 + w2)
        Δs = Δs_1
    else
        Δs = Δs_2
    end
    #Δs = combine((Δs_1, Δs_2))
    if val isa StochasticTriple
        Δs = combine((Δs, val.Δs))
    end
    return StochasticTriple{T}(value(val), delta(val), Δs)
end

function Base.getindex(
        C::AbstractArray, i1::StochasticTriple{T, V, FIs}, i2::Colon) where {T, V, FIs}
    [C[i1, i] for i in 1:size(C, 2)]
end
function Base.getindex(
        C::AbstractArray, i1::Colon, i2::StochasticTriple{T, V, FIs}) where {T, V, FIs}
    [C[i, i2] for i in 1:size(C, 1)]
end
function Base.getindex(C::AbstractArray, i1::Colon,
        i2::Array{<:StochasticTriple{T, V, FIs}}) where {T, V, FIs}
    return [C[i1, i] for i in i2]
end
function Base.getindex(C::AbstractArray, i1::Array{<:StochasticTriple{T, V, FIs}},
        i2::Colon) where {T, V, FIs}
    return [C[i, i2] for i in i1]
end