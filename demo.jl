include("_getdata.jl")

using MLJ
using ConformalPrediction
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

#pres = findall(isequal(true), Xy.presence)
#scatter!(Xy.longitude[pres], Xy.latitude[pres])

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