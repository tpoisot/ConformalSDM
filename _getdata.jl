import Downloads
import CSV
import Dates
using SpeciesDistributionToolkit
import SpeciesDistributionToolkit as SDT
using DataFrames
using CairoMakie

# Download the data if they don't exist
if ~isfile("data/occurrences.csv")
    Downloads.download("https://raw.githubusercontent.com/tpoisot/InterpretableSDMWithJulia/main/occurrences.csv", "data/occurrences.csv")
end

# Function to convert dates
bigdate = (s) -> Dates.Year(Dates.Date(s, Dates.dateformat"yyyy-mm-ddTH:M:SZ"))

# Get the places
polygons = [
    SDT.gadm("USA", "Oregon"),
    SDT.gadm("USA", "Washington"),
    SDT.gadm("USA", "California")
]
bbox = SDT._reconcile(SDT.boundingbox.(polygons; padding=1.0))

# Data
provider = RasterData(CHELSA2, BioClim)
opts = (; )
temperature = SDMLayer(provider, layer=1; opts..., bbox...)

# Filter by class of observation, only after 2000
sightings = CSV.File("data/occurrences.csv")
occ = [
    (record.longitude, record.latitude)
    for record in sightings
    if
    (record.classification == "Class A") &
    (bigdate(record.timestamp) >= Dates.Year(1900))
]
# Spatial filter
filter!(r -> bbox.bottom <= r[2] <= bbox.top, occ)
filter!(r -> bbox.left <= r[1] <= bbox.right, occ)

# Turn into occurrences
presencedata = Occurrences([Occurrence("Bigfoot", true, (r[1], r[2]), missing) for r in occ])

# Get the mask for all polygons
msks = [mask(temperature, p) for p in polygons]
temperature.indices = reduce(.|, [m.indices for m in msks])

# Presence layer
presence_layer = mask(temperature, presencedata)

# Background
possible_background = pseudoabsencemask(DistanceToEvent, presence_layer)

# Absence layer
absence_layer = backgroundpoints(
    nodata(possible_background, r -> r <= 20.0),
    3sum(presence_layer);
    replace=false
)

# Sanity check
f = Figure()
ax = Axis(f[1,1], aspect=DataAspect())
heatmap!(ax, temperature, colormap=:turbo)
[lines!(ax, p; color=:black) for p in polygons]
hidespines!(ax)
hidedecorations!(ax)
scatter!(ax, presence_layer, color=:white, strokecolor=:black, strokewidth=1, markersize=7)
current_figure()

# Cleanup
nodata!(absence_layer, false)
nodata!(presence_layer, false)
absence_layer = !absence_layer

y = vcat(
    values(presence_layer),
    values(absence_layer)
)

# Get the layers
L = [SDMLayer(provider; layer=i, opts..., bbox...) for i in 1:19]
for l in L
    l.x = temperature.x
    l.y = temperature.y
end
L = [mask(l, temperature) for l in L]
lnames = layers(provider)

tpls = []
for k in keys(L[1])
    pr = [presence_layer[k], absence_layer[k]]
    prcode = all(isnothing, pr) ? Inf : something(pr...)
    r = Dict(:latitude => northings(L[1])[k.I[1]], :longitude => eastings(L[1])[k.I[2]])
    r[:presence] = prcode
    for i in eachindex(lnames)
        r[Symbol(lnames[i])] = L[i][k]
    end
    push!(tpls, r)
end

# Get the full data
Xf = DataFrame(tpls)
Xy = Xf[findall(!isinf, Xf.presence),:]

# Get the future data
L = [SDMLayer(provider, Projection(SSP370, MRI_ESM2_0); layer=i, timespan=Dates.Year(2071) => Dates.Year(2100), opts..., bbox...) for i in 1:19]
for l in L
    l.x = temperature.x
    l.y = temperature.y
end
L = [mask(l, temperature) for l in L]

tpls = []
for k in keys(L[1])
    pr = [presence_layer[k], absence_layer[k]]
    prcode = all(isnothing, pr) ? Inf : something(pr...)
    r = Dict(:latitude => northings(L[1])[k.I[1]], :longitude => eastings(L[1])[k.I[2]])
    r[:presence] = prcode
    for i in eachindex(lnames)
        r[Symbol(lnames[i])] = L[i][k]
    end
    push!(tpls, r)
end

# Get the full data
fXf = dropmissing(DataFrame(tpls))