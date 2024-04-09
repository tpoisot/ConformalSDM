import ColorBlendModes

# Bivariate plot
n_stops = 5
rscl = 0.0:0.01:1.0

r1, r2 = pred, conformal

l1 = copy(r1)#rescale(r1, rscl);
l2 = copy(r2)#rescale(r2, rscl);

d1 = Int64.(round.((n_stops - 1) .* l1; digits = 0)) .+ 1
d2 = Int64.(round.((n_stops - 1) .* l2; digits = 0)) .+ 1

function bivariator(n1, n2)
    function bv(v1, v2)
        return n2 * (v2 - 1) + v1
    end
    return bv
end

b = bivariator(n_stops, n_stops).(d1, d2)
sort(unique(values(b)))
heatmap(b; colormap = :managua, colorrange = (1, n_stops * n_stops))

p0 = colorant"#e8e8e8ff"
p1 = colorant"#6c83b5ff"
p2 = colorant"#73ae80ff"
cm1 = LinRange(p0, p1, n_stops)
cm2 = LinRange(p0, p2, n_stops)
cmat = ColorBlendModes.BlendMultiply.(cm1, cm2')
cmap = vec(cmat)

f = Figure(; resolution = (700, 700))

m_biv = Axis(f[1, 1]; aspect = DataAspect())
heatmap!(m_biv, b; colormap = cmap, colorrange = (1, n_stops * n_stops))

m_v2 = Axis(f[2, 1]; aspect = DataAspect())
heatmap!(m_v2, d2; colormap = cm2, colorrange = (1, n_stops))

m_v1 = Axis(f[1, 2]; aspect = DataAspect())
heatmap!(m_v1, d1; colormap = cm1, colorrange = (1, n_stops))

m_leg = Axis(f[2, 2]; aspect = 1, xlabel = "Prediction", ylabel = "Certainty")
x = LinRange(minimum(r1), maximum(r1), n_stops)
y = LinRange(minimum(r2), maximum(r2), n_stops)
heatmap!(m_leg, x, y, reshape(1:(n_stops * n_stops), (n_stops, n_stops)); colormap = cmap)

current_figure()