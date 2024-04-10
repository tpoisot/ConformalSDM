include("_getdata.jl")

using MLJ
using ConformalPrediction
import ComputationalResources
using Shapley
using CairoMakie
CairoMakie.activate!(; px_per_unit = 2)

# Split the data to train a model
schema(Xy)
Xy.presence = coerce(Xy.presence, OrderedFactor)

y, X = unpack(select(Xy, Not([:longitude, :latitude])), ==(:presence); rng=420)

models(matching(X,y))

Tree = @load EvoTreeClassifier pkg=EvoTrees
tree = Tree(nbins=128, max_depth=4)

# Initial classifier
mach = machine(tree, X, y)
train, test = partition(eachindex(y), 0.7); # 70:30 split
fit!(mach, rows=train)
yhat = predict_mode(mach, X[test,:])
matthews_correlation(yhat, y[test])

# Get some prediction going
pred = similar(temperature)
pred.grid[findall(!isnothing, pred.grid)] .= pdf.(predict(mach, select(Xf, Not([:longitude, :latitude]))), true)
heatmap(pred, colormap=:navia)

# Conformal prediction with changing coverage
α = 0.05
conf_model = conformal_model(tree; coverage=1-α)
conf_mach = machine(conf_model, X, y)
fit!(conf_mach, rows=train)

# Make the uncertainty prediction
conformal = similar(temperature)
nopredict = similar(temperature)
conf_pred = predict(conf_mach, select(Xf, Not([:longitude, :latitude])))
conf_val = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in conf_pred]
conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in conf_pred]

conformal.grid[findall(!isnothing, conformal.grid)] .= conf_val
nopredict.grid[findall(!isnothing, nopredict.grid)] .= conf_false

f = Figure()
ax = Axis(f[2,1])
heatmap!(ax, pred, colormap=[:lightgrey, :lightgrey])
hm = heatmap!(ax, conformal, colormap=:navia, colorrange=(0,1))
Colorbar(f[1,1], hm; vertical=false)
ax2 = Axis(f[2,2])
heatmap!(ax2, pred, colormap=[:lightgrey, :lightgrey])
heatmap!(ax2, conformal, colormap=[:black, :black])
heatmap!(ax2, mask((conformal .> 0), (nopredict .> 0)), colormap=[:grey, :grey])
current_figure()

#pres = findall(!isequal(true), Xy.presence)
#scatter!(Xy.longitude[pres], Xy.latitude[pres])

VAR = :BIO8
idx = [i for i in 1:size(Xf, 1) if !isnothing(conformal[Xf.longitude[i], Xf.latitude[i]])]
B1 = shapley(x -> predict(conf_mach, x), Shapley.MonteCarlo(ComputationalResources.CPUThreads(32), 32), Xf[idx,:])

expvar = similar(pred)
expvar.grid[findall(!isnothing, expvar.grid)] .= Xf[:,VAR]

expl = similar(conformal)
expl.grid[findall(!isnothing, expl.grid)] .= pdf.(B1[VAR], true)

# Masks for the Shapley values
unsure_mask = mask((conformal .> 0), (nopredict .> 0))
sure_mask = conformal .> 0
sure_mask.grid[findall(!isnothing, nopredict.grid)] .= nothing

frange = (-0.3, 0.3)
f = Figure()
ax = Axis(f[2,1])
gl = f[2,2] = GridLayout()
hs = Axis(gl[1,2], xaxisposition=:top)
hu = Axis(gl[2,2])
es = Axis(gl[1,1], xaxisposition=:top)
eu = Axis(gl[2,1])
heatmap!(ax, pred, colormap=[:lightgrey, :lightgrey])
hm = heatmap!(ax, expl, colormap=:managua, colorrange=frange)
Colorbar(f[1,1], hm; vertical=false)
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