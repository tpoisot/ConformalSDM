include("_getdata.jl")

using MLJ
using ConformalPrediction
using ColorSchemes
using CairoMakie
using Statistics
using HypothesisTests
CairoMakie.activate!(; px_per_unit=2)

# Some info on color for the entire figures
mapbg = colorant"#ecebe8ee"
probacolor = ColorSchemes.navia
varcolor = ColorSchemes.bamako
rangecolor = ColorSchemes.ColorScheme(reverse(ColorSchemes.batlowW.colors)[15:end])
effectcolor = ColorSchemes.managua

# Split the data to train a model
#schema(Xy) # Check the type of the different inputs
Xy.presence = coerce(Xy.presence, OrderedFactor) # We turn the presence data into an ordered factor - false (absence) is the negative class

# Split the training data into features y and instances X - the coordinates are removed at this step
y, X = unpack(select(Xy, Not([:longitude, :latitude])), ==(:presence); rng=12345)
#models(matching(X, y)) # Verify the list of currently known compatible models

# The classifier we use is the EvoTreeClassifier, which is a BRT with some nice
# shortcuts to fit the histogram
Tree = @load EvoTreeClassifier pkg = EvoTrees

# Our default BRT will use 128 bins for the variable histogram, and stop trees to a depth of 6 - this has not been very rigorously tested but the model performance is good enough that there is no obvious need to change it
tree = Tree(nbins=128, max_depth=6)

# Forward variable selection for the BRT
include("_forwardselection.jl")
retained_variables = forwardselection(tree, X, y; rng=12345)
#retained_variables = [19, 10, 4, 14, 16] # Hardcoding these in so they save time
open("VARIABLESELECTION","w") do io
    show(io, retained_variables)
end

# We cut the orignal data (as well as the training data) to just use the selected variables
VARS = Symbol.("BIO" .* string.(retained_variables))
X = X[:, VARS]
Xp = select(Xf, Not([:latitude, :longitude]))[:, VARS]

# Initial classifier
mach = machine(tree, X, y)
fit!(mach)
evl_brt = evaluate(tree, X, y,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    measures=[f1score, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, balanced_accuracy, matthews_correlation],
    verbosity=0
)

# Save the BRT evaluation
function evalsave(str, evl)
    open(str,"w") do io
        show(io, MIME("text/plain"), evl)
    end
    return nothing
end
evalsave("00_eval_brt.txt", evl_brt)

# Also fit a thresholded model
point_tree = BinaryThresholdPredictor(tree, threshold=0.5)
point_mach = machine(point_tree, X, y) |> fit!
r = range(point_tree, :threshold, lower=0.1, upper=0.9)
tuned_point_predictor = TunedModel(
    point_tree,
    tuning=LatinHypercube(rng=12345),
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    range=r,
    measure=matthews_correlation,
    n=30,
)
tuned_mach = machine(tuned_point_predictor, X, y) |> fit!
threshold = report(tuned_mach).best_model.threshold

evl_tun = evaluate(report(tuned_mach).best_model, X, y,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    measures=[f1score, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, balanced_accuracy, matthews_correlation],
    verbosity=0
)
evalsave("01_eval_tuned.txt", evl_tun)

# Get some prediction going
pred = similar(temperature)
pred.grid[findall(!isnothing, pred.grid)] .= pdf.(predict(mach, Xp), true)
distrib = pred .>= threshold

fig_brt = Figure(resolution=(1200, 500))
ax_pred = Axis(fig_brt[2, 1], aspect=DataAspect())
ax_range = Axis(fig_brt[2, 2], aspect=DataAspect())
heatmap!(ax_pred, pred, colormap=[mapbg, mapbg])
hm = heatmap!(ax_pred, pred, colormap=probacolor, colorrange=(0.0, 1.0))
Colorbar(fig_brt[1, 1], hm; vertical=false, minorticksvisible=true, width=Relative(3.7 / 4))
heatmap!(ax_range, pred, colormap=[mapbg, mapbg])
heatmap!(ax_range, distrib, colormap=rangecolor)
legcol = [ColorSchemes.get(rangecolor, x, extrema(distrib)) for x in sort(unique(values(distrib)))]
leglab = ["absence", "presence"]
legbox = [PolyElement(color=c, strokecolor=:black, strokewidth=1) for c in legcol]
Legend(fig_brt[1, 2], legbox, leglab; orientation=:horizontal, tellheight=false, tellwidth=false, halign=:center, valign=:center, nbanks=1, framevisible=false)
current_figure()
save("01_brt_prediction.png", current_figure())

# Bagging of thr BRT to get an estimate of uncertainty due to data sampling
function getbag(n)
    train = rand(1:n, n)
    test = setdiff(1:n, train)
    return (train, test)
end
bag = [getbag(size(X,1)) for _ in Base.OneTo(100)]
evl_oob = evaluate(report(tuned_mach).best_model, X, y,
    resampling=bag,
    measures=[f1score, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, balanced_accuracy, matthews_correlation],
    verbosity=0
)
evalsave("02_eval_oob.txt", evl_oob)

ensemble = [fit!(machine(report(tuned_mach).best_model, X, y), rows=bag[i][1]) for i in 1:length(bag)]
outcome = convert(Matrix{Bool}, hcat([predict(component, Xp) for component in ensemble]...))
bsvar = similar(temperature)
bsvar.grid[findall(!isnothing, bsvar.grid)] .= vec(var(outcome, dims=2))

# Conformal prediction with a specific coverage rate
α = 0.05
conf_model = conformal_model(tree; coverage=1 - α)
conf_mach = machine(conf_model, X, y)
fit!(conf_mach)

evl_cnf = evaluate!(
    conf_mach,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    operation=predict,
    measure=[emp_coverage, ineff, size_stratified_coverage],
    verbosity = 0
)
evalsave("03_eval_conformal.txt", evl_cnf)

# Make the uncertainty prediction
conformal = similar(temperature)
confindex = findall(!isnothing, conformal.grid)
conforange = similar(temperature)
conf_pred = predict(conf_mach, Xp)
conf_true = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in conf_pred]
conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in conf_pred]
conf_either = [any(isnothing.([conf_true[i], conf_false[i]])) ? nothing : true for i in eachindex(conf_true)]
conformal.grid[findall(!isnothing, conformal.grid)] .= conf_true

conforange.grid[confindex[findall(!isnothing, conf_true)]] .= 3
conforange.grid[confindex[findall(!isnothing, conf_false)]] .= 0
# Split between things in/out of BRT range
conforange.grid[confindex[findall(!isnothing, conf_either)]] .= 1
inrange = keys(mask(distrib, mask(conforange .== 1, conforange)))
for ir in inrange
    conforange[ir] = 2
end

fig_variance = Figure(resolution=(1200, 500))
ax_bagvar = Axis(fig_variance[2, 1], aspect=DataAspect())
gl_varclass = fig_variance[2, 2] = GridLayout()
ax_spr = Axis(gl_varclass[2,1])
ax_upr = Axis(gl_varclass[2,2])
ax_uab = Axis(gl_varclass[1,2])
ax_sab = Axis(gl_varclass[1,1])
hm = heatmap!(ax_bagvar, bsvar, colormap=varcolor, colorrange=extrema(bsvar))
Colorbar(fig_variance[1, 1], hm; vertical=false, minorticksvisible=true, width=Relative(3.7 / 4))
legcol = [ColorSchemes.get(rangecolor, x, extrema(conforange)) for x in sort(unique(values(conforange)))]
leglab = ["absence", "uncertain (out)", "uncertain (in)", "presence"]
legbox = [PolyElement(color=c, strokecolor=:black, strokewidth=1) for c in legcol]
hist!(ax_sab, mask(conforange .== 0, bsvar), color=legcol[1], strokecolor=:grey, strokewidth=1, strokearound=true, bins=12)
hist!(ax_uab, mask(conforange .== 1, bsvar), color=legcol[2], strokecolor=:grey, strokewidth=1, strokearound=true, bins=12)
hist!(ax_upr, mask(conforange .== 2, bsvar), color=legcol[3], strokecolor=:grey, strokewidth=1, strokearound=true, bins=12)
hist!(ax_spr, mask(conforange .== 3, bsvar), color=legcol[4], strokecolor=:grey, strokewidth=1, strokearound=true, bins=12)
Legend(fig_variance[1, 2], legbox, leglab; orientation=:horizontal, tellheight=false, tellwidth=false, halign=:center, valign=:center, nbanks=1, framevisible=false)
for ax in [ax_sab, ax_uab, ax_upr, ax_spr]
    tightlimits!(ax)
    hideydecorations!(ax)
    hidespines!(ax)
end
current_figure()
save("06_bagging_variance.png", current_figure())

fig_conformal = Figure(resolution=(1200, 500))
ax_confpred = Axis(fig_conformal[2, 1], aspect=DataAspect())
ax_confrange = Axis(fig_conformal[2, 2], aspect=DataAspect())
heatmap!(ax_confpred, pred, colormap=[mapbg, mapbg])
hm = heatmap!(ax_confpred, conformal, colormap=probacolor, colorrange=(0.0, 1.0))
Colorbar(fig_conformal[1, 1], hm; vertical=false, minorticksvisible=true, width=Relative(3.7 / 4))
heatmap!(ax_confrange, conforange, colormap=rangecolor)
legcol = [ColorSchemes.get(rangecolor, x, extrema(conforange)) for x in sort(unique(values(conforange)))]
leglab = ["absence", "uncertain (out)", "uncertain (in)", "presence"]
legbox = [PolyElement(color=c, strokecolor=:black, strokewidth=1) for c in legcol]
Legend(fig_conformal[1, 2], legbox, leglab; orientation=:horizontal, tellheight=false, tellwidth=false, halign=:center, valign=:center, nbanks=1, framevisible=false)
current_figure()
save("02_conformal_prediction.png", current_figure())

# RCP projection to show uncertainty/novelty
novelty = convert(Bool, similar(temperature))
surfacearea = cellsize(pred)
for r in eachrow(fXf)
    novelty[r.longitude, r.latitude] = any([!(minimum(Xy[!,n]) <= r[n] <= maximum(Xy[!,n])) for n in VARS])
end

fconformal = similar(temperature)
fconforange = similar(temperature)
fconf_pred = predict(conf_mach, select(fXf, VARS))
for (i,pr) in enumerate(fconf_pred)
    fconformal[fXf.longitude[i], fXf.latitude[i]] = pdf(pr, true)
    fstat = 0
    if (pdf(pr, true) > 0) & (pdf(pr, false) > 0)
        fstat = 1
    end
    if (pdf(pr, true) > 0) & (pdf(pr, false) == 0)
        fstat = 2
    end
    fconforange[fXf.longitude[i], fXf.latitude[i]] = fstat
end

# Surface area

in_current = [sum(mask(replace(conforange .== i, false => nothing), surfacearea)) for i in [0,1,2,3]]
in_future = [sum(mask(replace(fconforange .== i, false => nothing), surfacearea)) for i in [0,1,2]]
in_novl = [sum(mask(replace(mask(novelty, fconforange .== i), false => nothing), surfacearea)) for i in [0,1,2]]
in_seen = [sum(mask(replace(mask(!novelty, fconforange .== i), false => nothing), surfacearea)) for i in [0,1,2]]

in_current ./= sum(in_current)
in_current = [in_current[1], sum(in_current[2:3]), in_current[4]]
in_future ./= sum(in_future)
in_novl ./= sum(in_novl)
in_seen ./= sum(in_seen)

open("05_rangesize.txt","w") do io
    println(io, "current")
    println(io,  in_current)
    println(io, "")
    println(io, "future")
    println(io,  in_future)
    println(io, "")
    println(io, "novel")
    println(io,  in_novl)
    println(io, "not novel")
    println(io,  in_seen)
    println(io, "")
    println(io, "test novel / seen")
    show(io, OneSampleTTest(in_novl .- in_seen))
    println(io, "")
    println(io, "test current / future")
    show(io, OneSampleTTest(in_current .- in_future))
end

# Level at which the pixel is included in the range
coverage_effect = DataFrame(α=Float64[], sure_presence=Float64[], unsure_presence=Float64[], unsure_absence=Float64[], sure_absence=Float64[], coverage=Float64[], ssc=Float64[], ineff=Float64[])
for α in LinRange(0.0, 0.25, 25)
    partial_conf_model = conformal_model(tree; coverage=1 - α)
    partial_conf_mach = machine(partial_conf_model, X, y)
    fit!(partial_conf_mach)
    evl = evaluate!(
        partial_conf_mach,
        resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
        operation=predict,
        measure=[emp_coverage, size_stratified_coverage, ineff]
    )
    partial_conf_pred = predict(partial_conf_mach, Xp)
    partial_conf_true = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in partial_conf_pred]
    partial_conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in partial_conf_pred]
    partial_conf_either = [any(isnothing.([partial_conf_true[i], partial_conf_false[i]])) ? nothing : true for i in eachindex(partial_conf_true)]
    # Make a map
    partial_conforange = similar(pred)
    partial_conforange.grid[confindex[findall(!isnothing, partial_conf_true)]] .= 3
    partial_conforange.grid[confindex[findall(!isnothing, partial_conf_false)]] .= 0
    # Split between things in/out of BRT range
    partial_conforange.grid[confindex[findall(!isnothing, partial_conf_either)]] .= 1
    partial_inrange = keys(mask(distrib, mask(partial_conforange .== 1, partial_conforange)))
    for ir in partial_inrange
        partial_conforange[ir] = 2
    end
    # Sure/unsure pixels update
    spr = count(isequal(3), partial_conforange.grid) == 0 ? 0.0 : sum(mask(partial_conforange .== 3, surfacearea))
    upr = count(isequal(2), partial_conforange.grid) == 0 ? 0.0 : sum(mask(partial_conforange .== 2, surfacearea))
    uab = count(isequal(1), partial_conforange.grid) == 0 ? 0.0 : sum(mask(partial_conforange .== 1, surfacearea))
    sab = count(isequal(0), partial_conforange.grid) == 0 ? 0.0 : sum(mask(partial_conforange .== 0, surfacearea))
    push!(coverage_effect, (α, spr, upr, uab, sab, evl.measurement[1], evl.measurement[2], evl.measurement[3]))
end

coverage_effect.total = coverage_effect.sure_presence .+ coverage_effect.unsure_presence .+ coverage_effect.unsure_absence

fig_risk = Figure(resolution=(1200, 500))
legcol = [ColorSchemes.get(rangecolor, x, extrema(conforange)) for x in sort(unique(values(conforange)))]
leglab = ["absence", "uncertain (out)", "uncertain (in)", "presence"]
ax_area = Axis(fig_risk[1:2, 1], yscale=sqrt, ylabel="Area (km²)", xlabel="Risk level (α)")
ax_cov = Axis(fig_risk[2, 2], ylabel="Coverage", xlabel="Risk level (α)", yaxisposition=:right)
ax_ineff = Axis(fig_risk[1, 2], ylabel="Inefficiency", xlabel=" ", yaxisposition=:right, xaxisposition=:top)
hlines!(ax_area, [1e-3 * sum(mask(distrib, cellsize(distrib)))], color=:black, linewidth=2, linestyle=:dash, label="BRT range")
lines!(ax_area, coverage_effect.α, 1e-3 .* coverage_effect.total, label="Total range", color=:grey, linewidth=4)
scatterlines!(ax_area, coverage_effect.α, 1e-3 .* coverage_effect.unsure_absence, label=leglab[2], color=:grey, markercolor=legcol[2], strokecolor=:black, strokewidth=1, linestyle=:dot, marker=:dtriangle)
scatterlines!(ax_area, coverage_effect.α, 1e-3 .* coverage_effect.unsure_presence, label=leglab[3], color=:grey, markercolor=legcol[3], strokecolor=:black, strokewidth=1, linestyle=:dot, marker=:utriangle)
scatterlines!(ax_area, coverage_effect.α, 1e-3 .* coverage_effect.sure_presence, label=leglab[4], color=:grey, markercolor=legcol[4], strokecolor=:black, strokewidth=1, linestyle=:dot)
scatterlines!(ax_cov, coverage_effect.α, coverage_effect.coverage, label="Empirical", color=:black)
scatterlines!(ax_cov, coverage_effect.α, coverage_effect.ssc, label="Size stratified", color=:black, marker=:utriangle)
scatterlines!(ax_ineff, coverage_effect.α, coverage_effect.ineff, color=:black)
axislegend(ax_area, nbanks=2)
axislegend(ax_cov)
linkxaxes!(ax_area, ax_cov)
linkxaxes!(ax_area, ax_ineff)
tightlimits!(ax_cov)
tightlimits!(ax_ineff)
current_figure()
save("03_coverage_effect.png", current_figure())

# Masks for the Shapley values based on certainty on the conformal predictor
sure_presence_mask = conforange .== 3
unsure_presence_mask = conforange .== 2
unsure_absence_mask = conforange .== 1
sure_absence_mask = conforange .== 0

# Shapley values calculation
include("_shapley.jl")
idx = [i for i in 1:size(Xf, 1) if !isnothing(pred[Xf.longitude[i], Xf.latitude[i]])]
prf = (x) -> predict(conf_mach, x)
ϕ = Dict()
for (i, v) in enumerate(VARS)
    @info "Performing explanations for variable $(v)"
    ϕ[v] = shap_list_points(prf, Xp, X, idx, i, 50)
end

# Shapley layers
positions = findall(!isnothing, sure_presence_mask.grid)
ψ = [similar(temperature) for _ in eachindex(VARS)]
for (i, v) in enumerate(VARS)
    ψ[i].grid[positions] .= pdf.(ϕ[v], true)
end

Wall = [sum(abs.(ψ[i])) for i in eachindex(VARS)]
Wspr = [sum(abs.(mask(sure_presence_mask, ψ[i]))) for i in eachindex(VARS)]
Wupr = [sum(abs.(mask(unsure_presence_mask, ψ[i]))) for i in eachindex(VARS)]
Wuab = [sum(abs.(mask(unsure_absence_mask, ψ[i]))) for i in eachindex(VARS)]
Wsab = [sum(abs.(mask(sure_absence_mask, ψ[i]))) for i in eachindex(VARS)]

Wall ./= sum(Wall)
Wspr ./= sum(Wspr)
Wupr ./= sum(Wupr)
Wuab ./= sum(Wuab)
Wsab ./= sum(Wsab)

vord = sortperm(Wall, rev=true)

fig_global = Figure(resolution=(1200, 500))
legcol = [ColorSchemes.get(rangecolor, x, extrema(conforange)) for x in sort(unique(values(conforange)))]
leglab = ["absence", "uncertain (out)", "uncertain (in)", "presence"]
ax_all = Axis(fig_global[1, 3], xticks=(1:length(VARS), string.(VARS[vord])), xticklabelrotation=π / 4, title="All predictions")
barplot!(ax_all, Wall[vord], color=:lightgrey, strokecolor=:black, strokewidth=1)

ax_spr = Axis(fig_global[1, 1], xticks=(1:length(VARS), string.(VARS[vord])), xticklabelrotation=π / 4, title="Sure presences")
barplot!(ax_spr, Wspr[vord], color=legcol[4], strokecolor=:black, strokewidth=1)
ax_sab = Axis(fig_global[2, 1], xticks=(1:length(VARS), string.(VARS[vord])), xticklabelrotation=π / 4, title="Sure absences")
barplot!(ax_sab, Wsab[vord], color=legcol[1], strokecolor=:black, strokewidth=1)

ax_upr = Axis(fig_global[1, 2], xticks=(1:length(VARS), string.(VARS[vord])), xticklabelrotation=π / 4, title="Uncertain presences")
barplot!(ax_upr, Wupr[vord], color=legcol[3], strokecolor=:black, strokewidth=1)
ax_uab = Axis(fig_global[2, 2], xticks=(1:length(VARS), string.(VARS[vord])), xticklabelrotation=π / 4, title="Uncertain absences")
barplot!(ax_uab, Wuab[vord], color=legcol[2], strokecolor=:black, strokewidth=1)

legbox = [PolyElement(color=c, strokecolor=:black, strokewidth=1) for c in legcol]
Legend(fig_global[2, 3], legbox, leglab; orientation=:horizontal, tellheight=false, tellwidth=false, halign=:center, valign=:center, nbanks=2, framevisible=false)

for ax in [ax_spr, ax_upr, ax_uab, ax_sab]
    scatterlines!(ax, Wall[vord], color=:red, markersize=15, marker=:star4, linestyle=:dot)
    linkyaxes!(ax_all, ax)
    linkxaxes!(ax_all, ax)
    ylims!(ax, low=0.0)
end
ylims!(ax_all, low=0.0)
xlims!(ax_all, 0, length(VARS) + 1)
current_figure()
save("04_variable_global_importance.png", current_figure())

# Plot the explanations for the most important variable
V = VARS[1]
D = layerdescriptions(provider)[string(V)]
vx = Xp[:, V]
vy = pdf.(ϕ[V], true)

expvar = similar(pred)
expvar.grid[findall(!isnothing, expvar.grid)] .= vx

expl = similar(pred)
expl.grid[findall(!isnothing, expl.grid)] .= vy

frange = maximum(abs.(extrema(vy))) .* (-1, 1)

fig_shapley = Figure(resolution=(1200, 500))
gl = fig_shapley[1, 1] = GridLayout()
legcol = [ColorSchemes.get(rangecolor, x, extrema(conforange)) for x in sort(unique(values(conforange)))]
leglab = ["absence", "uncertain (out)", "uncertain (in)", "presence"]

spl = Axis(gl[2, 1], xaxisposition=:bottom, xlabel=D, ylabel="Effect on average prediction")
phs = Axis(gl[1, 1])
ehs = Axis(gl[2, 2])
legbox = [PolyElement(color=c, strokecolor=:black, strokewidth=1) for c in legcol]
Legend(gl[1, 2], legbox, leglab; orientation=:horizontal, tellheight=false, tellwidth=false, halign=:center, valign=:center, nbanks=2, framevisible=false)

density!(ehs, mask(sure_absence_mask, expl), color=(legcol[1], 0.8), direction=:y, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(ehs, mask(unsure_absence_mask, expl), color=(legcol[2], 0.8), direction=:y, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(ehs, mask(unsure_presence_mask, expl), color=(legcol[3], 0.8), direction=:y, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(ehs, mask(sure_presence_mask, expl), color=(legcol[4], 0.8), direction=:y, strokecolor=:grey, strokewidth=1, strokearound=true)

density!(phs, mask(sure_absence_mask, expl), color=(legcol[1], 0.8), direction=:x, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(phs, mask(unsure_absence_mask, expl), color=(legcol[2], 0.8), direction=:x, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(phs, mask(unsure_presence_mask, expl), color=(legcol[3], 0.8), direction=:x, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(phs, mask(sure_presence_mask, expl), color=(legcol[4], 0.8), direction=:x, strokecolor=:grey, strokewidth=1, strokearound=true)

scatter!(spl, mask(sure_absence_mask, expvar), mask(sure_absence_mask, expl), color=legcol[1], markersize=3, transparency=0.5)
scatter!(spl, mask(unsure_absence_mask, expvar), mask(unsure_absence_mask, expl), color=legcol[2], markersize=3, transparency=0.5)
scatter!(spl, mask(unsure_presence_mask, expvar), mask(unsure_presence_mask, expl), color=legcol[3], markersize=3, transparency=0.5)
scatter!(spl, mask(sure_presence_mask, expvar), mask(sure_presence_mask, expl), color=legcol[4], markersize=3, transparency=0.5)

for hax in [spl, phs, ehs]
    tightlimits!(hax)
end
hidedecorations!(ehs)
hidedecorations!(phs)
hidespines!(ehs)
hidespines!(phs)
colgap!(gl, 0)
rowgap!(gl, 0)
colsize!(gl, 1, Relative(0.7))
rowsize!(gl, 2, Relative(0.7))
current_figure()
save("05_local_importance.png", current_figure())
