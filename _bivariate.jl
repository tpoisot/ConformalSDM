import ColorBlendModes

# Bivariate plot
n_stops = 3
rscl = 0.0:0.01:1.0

r1, r2 = fconforange, conforange

l1 = rescale(r1, (0., 1.))#rescale(r1, rscl);
l2 = rescale(r2, (0., 1.))#rescale(r2, rscl);

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
heatmap(b; colormap = :Spectral)#, colorrange = (1, n_stops * n_stops))

p0 = colorant"#e8e8e8ff"
p1 = colorant"#6c83b5ff"
p2 = colorant"#73ae80ff"

cm1 = LinRange(p0, p1, n_stops)
cm2 = LinRange(p0, p2, n_stops)
cm1 = [colorant"#F5EFE6", colorant"#E8DFCA", colorant"#4F6F52"]
cm2 = [colorant"#C40C0C", colorant"#E8DFCA", colorant"#4F6F52"]
cmat = ColorBlendModes.BlendMultiply.(cm1, cm2')

cmat[1,1] = colorant"#E8DFCA"
cmat[2,2] = colorant"#E8DFCA"
cmat[3,3] = colorant"#E8DFCA"

cmat[3,1] = colorant"#1A4D2E" # sure gain
cmat[1,3] = colorant"#DD5746" # sure loss

cmat[3,2] = colorant"#FFC470" # possible loss
cmat[2,3] = colorant"#FFC470" # possible loss

cmat[1,2] = colorant"#4793AF" # possible gain
cmat[2,1] = colorant"#4793AF" # possible gain

cmat

cmap = vec(cmat)

f = Figure(resolution=(1200, 500))

m_biv = Axis(f[1, 1]; aspect = DataAspect())
heatmap!(m_biv, b; colormap = cmap, colorrange = (1, n_stops * n_stops))

# m_v2 = Axis(f[2, 1]; aspect = DataAspect())
# heatmap!(m_v2, d2; colormap = cm2, colorrange = (1, n_stops))
# 
# m_v1 = Axis(f[1, 2]; aspect = DataAspect())
# heatmap!(m_v1, d1; colormap = cm1, colorrange = (1, n_stops))

m_leg = Axis(f[1, 2]; aspect = 1, xlabel = "Future range", ylabel = "Current range")
x = LinRange(minimum(r1), maximum(r1), n_stops)
y = LinRange(minimum(r2), maximum(r2), n_stops)
heatmap!(m_leg, x, y, reshape(1:(n_stops * n_stops), (n_stops, n_stops)); colormap = cmap)

current_figure()