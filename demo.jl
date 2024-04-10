include("_getdata.jl")

using MLJ
using ConformalPrediction
using Shapley
using Shapley.ComputationalResources
using CairoMakie
CairoMakie.activate!(; px_per_unit=2)

# Split the data to train a model
schema(Xy)
Xy.presence = coerce(Xy.presence, OrderedFactor)

y, X = unpack(select(Xy, Not([:longitude, :latitude])), ==(:presence); rng=420)

models(matching(X, y))

Tree = @load EvoTreeClassifier pkg = EvoTrees
tree = Tree(nbins=128, max_depth=6)

# Initial classifier
mach = machine(tree, X, y)
fit!(mach)
evaluate(tree, X, y,
    resampling=CV(shuffle=true),
    measures=[f1score, false_discovery_rate, balanced_accuracy, matthews_correlation],
    verbosity=0
)

# Get some prediction going
pred = similar(temperature)
pred.grid[findall(!isnothing, pred.grid)] .= pdf.(predict(mach, select(Xf, Not([:longitude, :latitude]))), true)
heatmap(pred, colormap=:navia)

# Conformal prediction with changing coverage
α = 0.05
conf_model = conformal_model(tree; coverage=1 - α)
conf_mach = machine(conf_model, X, y)
fit!(conf_mach)

evaluate!(
    conf_mach,
    resampling=CV(nfolds=10, shuffle=true; rng=123),
    operation=predict,
    measure=[emp_coverage, ssc]
)

# Make the uncertainty prediction
conformal = similar(temperature)
nopredict = similar(temperature)
conf_pred = predict(conf_mach, select(Xf, Not([:longitude, :latitude])))
conf_val = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in conf_pred]
conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in conf_pred]

conformal.grid[findall(!isnothing, conformal.grid)] .= conf_val
nopredict.grid[findall(!isnothing, nopredict.grid)] .= conf_false

bgc = colorant"#ecebe8ee"

f = Figure(resolution=(1200, 500))
ax = Axis(f[2, 1], aspect=DataAspect())
heatmap!(ax, pred, colormap=[bgc, bgc])
hm = heatmap!(ax, conformal, colormap=:navia, colorrange=(0.0, 1.0))
Colorbar(f[1, 1], hm; vertical=false, minorticksvisible=true, width=Relative(3.7/4))
ax2 = Axis(f[2, 2], aspect=DataAspect())
heatmap!(ax2, pred, colormap=[bgc, bgc])
heatmap!(ax2, conformal, colormap=[:black, :black])
heatmap!(ax2, mask((conformal .> 0), (nopredict .> 0)), colormap=[:grey, :grey])
current_figure()

#pres = findall(isequal(true), Xy.presence)
#scatter!(Xy.longitude[pres], Xy.latitude[pres], markersize=2, color=:teal)
#current_figure()

idx = [i for i in 1:size(Xf, 1) if !isnothing(conformal[Xf.longitude[i], Xf.latitude[i]])]
B1 = shapley(x -> predict(conf_mach, x), Shapley.MonteCarlo(CPUThreads(), 32), Xf[idx, :])

VAR = :BIO1
expvar = similar(pred)
expvar.grid[findall(!isnothing, expvar.grid)] .= Xf[:, VAR]

expl = similar(conformal)
expl.grid[findall(!isnothing, expl.grid)] .= pdf.(B1[VAR], true)

# Masks for the Shapley values
unsure_mask = mask((conformal .> 0), (nopredict .> 0))
sure_mask = conformal .> 0
sure_mask.grid[findall(!isnothing, nopredict.grid)] .= nothing

frange = (-0.4, 0.4)
f = Figure()
ax = Axis(f[2, 1])
gl = f[2, 2] = GridLayout()
hs = Axis(gl[1, 2], xaxisposition=:top)
hu = Axis(gl[2, 2])
es = Axis(gl[1, 1], xaxisposition=:top)
eu = Axis(gl[2, 1])
heatmap!(ax, pred, colormap=[:lightgrey, :lightgrey])
hm = heatmap!(ax, expl, colormap=:managua, colorrange=frange)
Colorbar(f[1, 1], hm; vertical=false)
hist!(hs, mask(sure_mask, expl), color=:black, bins=40, direction=:x)
hist!(hu, mask(unsure_mask, expl), color=:grey, bins=40, direction=:x)
scatter!(es, mask(sure_mask, expvar), mask(sure_mask, expl), color=:black)
scatter!(eu, mask(unsure_mask, expvar), mask(unsure_mask, expl), color=:grey)
for hax in [es, eu, hs, hu]
    tightlimits!(hax)
    ylims!(hax, frange...)
end
hidedecorations!(hs)
hidedecorations!(hu)
hidespines!(hs)
hidespines!(hu)
linkxaxes!(es, eu)
colgap!(gl, 0)
rowgap!(gl, 0)
colsize!(f.layout, 1, Relative(0.45))
rowsize!(f.layout, 1, Relative(0.01))
colsize!(gl, 1, Relative(0.85))
current_figure()

#=
# Get the Shapley values
idx = [i for i in 1:size(Xf, 1) if !isnothing(conformal[Xf.longitude[i], Xf.latitude[i]])]
B1 = shapley(x -> predict(conf_mach, x), Shapley.MonteCarlo(16), Xf[idx,:], :BIO1)

expl = similar(conformal)
expl.grid[findall(!isnothing, expl.grid)] .= pdf.(B1, true)

f = Figure()
ax = Axis(f[1,1])
heatmap!(ax, pred, colormap=[:lightgrey, :lightgrey])
hm = heatmap!(ax, expl, colormap=:managua, colorrange=(-0.3, 0.3))
Colorbar(f[1,2], hm)
current_figure()

scatter(Xf.BIO1[idx], values(expl))
=#