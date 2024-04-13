include("_getdata.jl")

using MLJ
using ConformalPrediction
using CairoMakie
CairoMakie.activate!(; px_per_unit=2)

# Background color for the maps - this is the color for NODATA
bgc = colorant"#ecebe8ee"

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
#retained_variables = forwardselection(tree, X, y; rng=12345)
retained_variables = [19, 4, 10, 12, 8, 18] # Hardcoding these in so they save time

# We cut the orignal data (as well as the training data) to just use the selected variables
VARS = Symbol.("BIO" .* string.(retained_variables))
X = X[:, VARS]
Xp = select(Xf, Not([:latitude, :longitude]))[:, VARS]

# Initial classifier
mach = machine(tree, X, y)
fit!(mach)
evaluate(tree, X, y,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    measures=[f1score, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, balanced_accuracy, matthews_correlation],
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

evaluate(report(tuned_mach).best_model, X, y,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    measures=[f1score, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, balanced_accuracy, matthews_correlation],
    verbosity=0
)

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
coverage_effect = DataFrame(α=Float64[], sure=Float64[], total=Float64[], coverage=Float64[], ssc=Float64[], ineff=Float64[])
surfacearea = cellsize(pred)
for α in LinRange(0.0, 0.25, 20)
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
ax = Axis(f[1:2, 1], yscale=sqrt, ylabel="Area (km²)", xlabel="Risk level (α)")
ax2 = Axis(f[2, 2], ylabel="Coverage", xlabel="Risk level (α)",yaxisposition=:right)
ax3 = Axis(f[1, 2], ylabel="Inefficiency", xlabel=" ",yaxisposition=:right, xaxisposition=:top)
scatterlines!(ax, coverage_effect.α, 1e-3 .* coverage_effect.total, label="Total range", color=:grey, linestyle=:dash)
scatterlines!(ax, coverage_effect.α, 1e-3 .* coverage_effect.sure, label="Certain range", color=:black)
scatterlines!(ax2, coverage_effect.α, coverage_effect.coverage, label="Empirical", color=:black)
scatterlines!(ax2, coverage_effect.α, coverage_effect.ssc, label="Size stratified", color=:black, marker=:utriangle)
scatterlines!(ax3, coverage_effect.α, coverage_effect.ineff, color=:black)
axislegend(ax)
axislegend(ax2)
linkxaxes!(ax, ax2)
linkxaxes!(ax, ax3)
tightlimits!(ax2)
tightlimits!(ax3)
current_figure()
save("03_coverage_effect.png", current_figure())

# Masks for the Shapley values
unsure_mask = mask((conformal .> 0), (nopredict .> 0))
sure_mask = conformal .> 0
sure_mask.grid[findall(!isnothing, nopredict.grid)] .= nothing

# Shapley values calculation
include("_shapley.jl")
idx = [i for i in 1:size(Xf, 1) if !isnothing(conformal[Xf.longitude[i], Xf.latitude[i]])]
prf = (x) -> predict(conf_mach, x)
ϕ = Dict()
for (i, v) in enumerate(VARS)
    # This is a very long operation -- the Shapley code has been optimized somewhat but there is no alternative to making n predictions for each variable for each point, which scales up to hundreds of millions of model runs very rapidly
    @info "Performing explanations for variable $(v)"
    ϕ[v] = shap_list_points(prf, Xp, X, idx, i, 50)
end

# Assess the importance of variables - this is only done for the pixels where the conformal model predicts the presence, either alone or as part of a set that also contains the absence
W = sum.([abs.(pdf.(ϕ[v], true)) for v in VARS])
W ./= sum(W)

# Get the variable importance for the sure/unsure part of the range
all_pos = findall(!isnothing, conformal)
sure_pos = findall(!isnothing, sure_mask)
unsure_pos = findall(!isnothing, unsure_mask)
sure_idx = indexin(sure_pos, all_pos)
unsure_idx = indexin(unsure_pos, all_pos)

Ws = sum.([abs.(pdf.(ϕ[v][sure_idx], true)) for v in VARS])
Ws ./= sum(Ws)

Wu = sum.([abs.(pdf.(ϕ[v][unsure_idx], true)) for v in VARS])
Wu ./= sum(Wu)

f = Figure(resolution=(1200, 500))
ax1 = Axis(f[1,1], xticks= (1:length(VARS), string.(VARS)), xticklabelrotation=π/4, title="Range")
barplot!(ax1, W, color=:grey)
ax2 = Axis(f[1,2], xticks= (1:length(VARS), string.(VARS)), xticklabelrotation=π/4, title="Certain")
barplot!(ax2, Ws, color=:grey)
ax3 = Axis(f[1,3], xticks= (1:length(VARS), string.(VARS)), xticklabelrotation=π/4, title="Uncertain")
barplot!(ax3, Wu, color=:grey)
for ax in [ax2, ax3]
    scatter!(ax, W, color=:black, markersize=15)
end
linkyaxes!(ax2, ax1)
linkyaxes!(ax2, ax3)
for ax in [ax2, ax3, ax1]
    ylims!(ax, low=0.0)
    xlims!(ax, 0, length(VARS)+1)
end
current_figure()
save("04_variable_global_importance.png", current_figure())

# Plot the explanations for the most important variable
V = VARS[1]
D = layerdescriptions(provider)[string(V)]
vx = Xp[:, V]
vy = pdf.(ϕ[V], true)

expvar = similar(pred)
expvar.grid[findall(!isnothing, expvar.grid)] .= vx

expl = similar(conformal)
expl.grid[findall(!isnothing, expl.grid)] .= vy

frange = maximum(abs.(extrema(vy))) .* (-1, 1)
f = Figure(resolution=(1200, 500))
gl = f[1, 1] = GridLayout()
spl = Axis(gl[2, 1], xaxisposition=:bottom, xlabel=D, ylabel="Effect on average prediction")
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
rowsize!(gl, 2, Relative(0.5))
current_figure()
save("05_local_importance.png", current_figure())