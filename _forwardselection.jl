function forwardselection(model, X, y; rng=12345, verbose=true)
    pool = collect(1:size(X, 2))
    retained_variables = eltype(pool)[]
    mcc_to_beat = -Inf
    improved = true
    while improved
        improved = false
        mcc_this_round = zeros(Float64, length(pool))
        for (i, v) in enumerate(pool)
            thissel = [retained_variables..., v]
            ev = evaluate(model, X[:, thissel], y; verbosity=0, resampling=StratifiedCV(nfolds=10, shuffle=true, rng=rng), measures=[matthews_correlation])
            mcc_this_round[i] = first(ev.measurement)
        end
        if maximum(mcc_this_round) >= mcc_to_beat
            improved = true
            mcc_to_beat, selvar = findmax(mcc_this_round)
            push!(retained_variables, popat!(pool, selvar))
        end
        if verbose
            @info "Included variable $(last(retained_variables)) - MCC ≈ $(round(mcc_to_beat; digits=4))"
        end
    end
    return retained_variables
end