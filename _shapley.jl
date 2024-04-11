function mcsample(X, Z, i, j, n)
    # Initial sample matrix
    ξ = similar(X, n)
    for row in axes(ξ, 1)
        ξ[row, :] = X[i, :]
    end
    # Observations for boostrap
    p = rand(axes(Z, 1), n)
    # And now we shuffle
    shuffling = rand(Bool, size(ξ))
    for ob in axes(shuffling, 1)
        for va in axes(shuffling, 2)
            if va != j
                if shuffling[ob, va]
                    ξ[ob, va] = Z[p[ob], va]
                end
            end
        end
    end
    ζ = copy(ξ)
    ζ[:, j] .= Z[p, j]
    return ξ, ζ
end

mcsample(X, i, j, n) = mcsample(X, X, i, j, n)

function shap_one_point(f, X, Z, i, j, n)
    ξ, ζ = mcsample(X, Z, i, j, n)
    ϕ = f(ξ) - f(ζ)
    return sum(ϕ)/length(ϕ)
end

function shap_list_points(f, X, Z, i, j, n)
    p0 = shap_one_point(f, X, Z, first(i), j, n)
    vals = Vector{typeof(p0)}(undef, length(i))
    vals[begin] = p0
    Threads.@threads for x in 2:length(i)
        vals[x] = shap_one_point(f, X, Z, i[x], j, n))
    end
    return vals
end

function shap_all_points(f, X, Z, j, n)
    return shap_list_points(f, X, Z, axes(X, 1), j, n)
end

#v2 = shap_all_points(prf, select(Xf, Not([:longitude, :latitude])), X, 2, 50)

