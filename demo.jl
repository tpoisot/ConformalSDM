include("_getdata.jl")

using MLJ
using ConformalPrediction
using CairoMakie
CairoMakie.activate!(; px_per_unit=2)

bgc = colorant"#ecebe8ee"

# Split the data to train a model
schema(Xy)
Xy.presence = coerce(Xy.presence, OrderedFactor)

y, X = unpack(select(Xy, Not([:longitude, :latitude])), ==(:presence); rng=12345)
#models(matching(X, y))

Tree = @load EvoTreeClassifier pkg = EvoTrees
tree = Tree(nbins=128, max_depth=6)

# Forward variable selection for the BRT
include("_forwardselection.jl")
#retained_variables = forwardselection(tree, X, y; rng=12345)
retained_variables = [19, 4, 10, 12, 8, 18] # Hardcoding these in so they save time

# Selected variables
VARS = Symbol.("BIO" .* string.(retained_variables))
X = X[:, VARS]
Xp = select(Xf, Not([:latitude, :longitude]))[:, VARS]

# Initial classifier
mach = machine(tree, X, y)
fit!(mach)
evaluate(tree, X, y,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    measures=[f1score, false_discovery_rate, balanced_accuracy, matthews_correlation],
    verbosity=0
)

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

# Get some prediction going
pred = similar(temperature)
pred.grid[findall(!isnothing, pred.grid)] .= pdf.(predict(mach, Xp), true)
distrib = pred .>= threshold

f = Figure(resolution=(1200, 500))
ax = Axis(f[2, 1], aspect=DataAspect())
heatmap!(ax, pred, colormap=[bgc, bgc])
hm = heatmap!(ax, pred, colormap=:navia, colorrange=(0.0, 1.0))
Colorbar(f[1, 1], hm; vertical=false, minorticksvisible=true, width=Relative(3.7 / 4))
ax2 = Axis(f[2, 2], aspect=DataAspect())
heatmap!(ax2, pred, colormap=[bgc, bgc])
heatmap!(ax2, distrib, colormap=[:transparent, :black])
current_figure()
save("01_brt_prediction.png", current_figure())

# Conformal prediction with a specific coverage rate
α = 0.05
conf_model = conformal_model(tree; coverage=1 - α)
conf_mach = machine(conf_model, X, y)
fit!(conf_mach)

evaluate!(
    conf_mach,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    operation=predict,
    measure=[emp_coverage, ineff, size_stratified_coverage]
)

# Make the uncertainty prediction
conformal = similar(temperature)
nopredict = similar(temperature)
conf_pred = predict(conf_mach, Xp)
conf_val = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in conf_pred]
conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in conf_pred]

conformal.grid[findall(!isnothing, conformal.grid)] .= conf_val
nopredict.grid[findall(!isnothing, nopredict.grid)] .= conf_false

f = Figure(resolution=(1200, 500))
ax = Axis(f[2, 1], aspect=DataAspect())
heatmap!(ax, pred, colormap=[bgc, bgc])
hm = heatmap!(ax, conformal, colormap=:navia, colorrange=(0.0, 1.0))
Colorbar(f[1, 1], hm; vertical=false, minorticksvisible=true, width=Relative(3.7 / 4))
ax2 = Axis(f[2, 2], aspect=DataAspect())
heatmap!(ax2, pred, colormap=[bgc, bgc])
heatmap!(ax2, conformal, colormap=[:black, :black])
heatmap!(ax2, mask((conformal .> 0), (nopredict .> 0)), colormap=[:grey, :grey])
current_figure()
save("02_conformal_prediction.png", current_figure())

# Level at which the pixel is included in the range
coverage_effect = DataFrame(α=Float64[], sure=Float64[], total=Float64[], coverage=Float64[], ssc=Float64[])
surfacearea = cellsize(pred)
for α in LinRange(0.0, 0.25, 8)
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
    partial_conf_val = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in partial_conf_pred]
    partial_conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in partial_conf_pred]
    # Pixels that are true only
    sure_pixels = setdiff(findall(!isnothing, partial_conf_val), findall(!isnothing, partial_conf_false))
    unsure_pixels = findall(!isnothing, partial_conf_val)
    # Sure/unsure pixels update
    s = !isempty(sure_pixels) ? sum(surfacearea.grid[findall(!isnothing, surfacearea.grid)[sure_pixels]]) : 0.0
    u = !isempty(unsure_pixels) ? sum(surfacearea.grid[findall(!isnothing, surfacearea.grid)[unsure_pixels]]) : 0.0
    push!(coverage_effect, (α, s, u, evl.measurement[1], evl.measurement[2], evl.measurement[3]))
end

f = Figure(resolution=(1200, 500))
ax = Axis(f[1, 1], yscale=sqrt, ylabel="Area (km²)", xlabel="Risk level (α)")
ax2 = Axis(f[1, 2], ylabel="Coverage", xlabel="Risk level (α)")
scatterlines!(ax, coverage_effect.α, 1e-3 .* coverage_effect.sure, label="Certain range", color=:black)
scatterlines!(ax, coverage_effect.α, 1e-3 .* coverage_effect.total, label="Total range", color=:grey, linestyle=:dash)
scatter!(ax2, coverage_effect.α, coverage_effect.coverage, label="Empirical", color=:black)
scatter!(ax2, coverage_effect.α, coverage_effect.ssc, label="Size stratified", color=:black, marker=:diamond)
axislegend(ax)
axislegend(ax2)
current_figure()
save("03_coverage_effect.png", current_figure())

# Masks for the Shapley values
unsure_mask = mask((conformal .> 0), (nopredict .> 0))
sure_mask = conformal .> 0
sure_mask.grid[findall(!isnothing, nopredict.grid)] .= nothing

#pres = findall(isequal(true), Xy.presence)
#scatter!(Xy.longitude[pres], Xy.latitude[pres], markersize=2, color=:teal)
#current_figure()

include("_shapley.jl")
idx = [i for i in 1:size(Xf, 1) if !isnothing(conformal[Xf.longitude[i], Xf.latitude[i]])]
prf = (x) -> predict(conf_mach, x)
ϕ = Dict()
for (i, v) in enumerate(VARS)
    @info "$(v) done"
    ϕ[v] = shap_list_points(prf, Xp, X, idx, i, 50)
end

W = sum.([abs.(pdf.(ϕ[v], true)) for v in VARS])
W ./= sum(W)

V = VARS[1]
D = layerdescriptions(provider)[string(V)]
vx = Xp[:, V]
vy = pdf.(ϕ[V], true)

expvar = similar(pred)
expvar.grid[findall(!isnothing, expvar.grid)] .= vx

expl = similar(conformal)
expl.grid[findall(!isnothing, expl.grid)] .= vy

frange = maximum(abs.(extrema(vy))) .* (-1, 1)
f = Figure()
gl = f[1, 1] = GridLayout()
spl = Axis(gl[2, 1], xaxisposition=:bottom, xlabel=D)
phs = Axis(gl[1, 1])
ehs = Axis(gl[2, 2])
gl2 = gl[1, 2] = GridLayout()
ax = Axis(gl2[1, 1], aspect=DataAspect())
heatmap!(ax, pred, colormap=[bgc, bgc])
hm = heatmap!(ax, expl, colormap=:managua, colorrange=frange)
Colorbar(gl2[1, 2], hm; vertical=true, height=Relative(0.82))
density!(ehs, mask(unsure_mask, expl), color=(:grey, 0.2), direction=:y, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(ehs, mask(sure_mask, expl), color=(:black, 0.5), direction=:y, strokecolor=:black, strokewidth=1, strokearound=true)
density!(phs, mask(unsure_mask, expvar), color=(:grey, 0.2), direction=:x, strokecolor=:grey, strokewidth=1, strokearound=true)
density!(phs, mask(sure_mask, expvar), color=(:black, 0.5), direction=:x, strokecolor=:black, strokewidth=1, strokearound=true)
scatter!(spl, mask(unsure_mask, expvar), mask(unsure_mask, expl), color=:grey, markersize=1, transparency=0.5)
scatter!(spl, mask(sure_mask, expvar), mask(sure_mask, expl), color=:black, markersize=2, transparency=0.5)
for hax in [spl, phs, ehs]
    tightlimits!(hax)
end
hidedecorations!(ax)
hidedecorations!(ehs)
hidedecorations!(phs)
hidespines!(ehs)
hidespines!(phs)
linkxaxes!(spl, phs)
linkyaxes!(spl, ehs)
colgap!(gl, 0)
rowgap!(gl, 0)
colsize!(gl, 1, Relative(0.6))
rowsize!(gl, 2, Relative(0.6))
current_figure()
