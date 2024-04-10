
sureat = similar(temperature)
for α in reverse(LinRange(0.0001, 0.1, 50))
    @info α
    conf_model = conformal_model(tree; coverage=1-α)
    conf_mach = machine(conf_model, X, y)
    fit!(conf_mach, rows=train)
    conf_pred = predict(conf_mach, select(Xf, Not([:longitude, :latitude])))
    conf_true = [ismissing(p) ? nothing : (pdf(p, true) == 0 ? nothing : pdf(p, true)) for p in conf_pred]
    conf_false = [ismissing(p) ? nothing : (pdf(p, false) == 0 ? nothing : pdf(p, false)) for p in conf_pred]
    pos = findall((.!isnothing.(conf_true)) .& (isnothing.(conf_false)))
    sureat.grid[findall(!isnothing, sureat.grid)[pos]] .= 1-α
end

replace!(sureat, 0.0 => nothing)

f = Figure()
ax = Axis(f[2,1])
heatmap!(ax, pred, colormap=[:lightgrey, :lightgrey])
hm = heatmap!(ax, sureat, colormap=:lipari, colorrange=(0.9,1.0))
Colorbar(f[1,1], hm; vertical=false)
current_figure()