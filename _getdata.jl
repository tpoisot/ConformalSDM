import Downloads
import CSV
import Dates
using SpeciesDistributionToolkit
using DataFrames

# Download the data if they don't exist
if ~isfile("data/occurrences.csv")
    Downloads.download("https://raw.githubusercontent.com/tpoisot/InterpretableSDMWithJulia/main/occurrences.csv", "data/occurrences.csv")
end

# Function to convert dates
bigdate = (s) -> Dates.Year(Dates.Date(s, Dates.dateformat"yyyy-mm-ddTH:M:SZ"))

# Filter by class of observation, only after 2000
sightings = CSV.File("data/occurrences.csv")
occ = [
    (record.longitude, record.latitude)
    for record in sightings
    if
    (record.classification == "Class A") &
    (bigdate(record.timestamp) >= Dates.Year(1900))
]
filter!(r -> -90 <= r[2] <= 90, occ)
filter!(r -> -180 <= r[1] <= 180, occ)

# Spatial filter
filter!(r -> 30 <= r[2] <= 52, occ)
filter!(r -> -130 <= r[1] <= -110, occ)

# Bounding box
boundingbox = (
    left=minimum(first.(occ)),
    right=maximum(first.(occ)),
    bottom=minimum(last.(occ)),
    top=maximum(last.(occ)),
)

# Data
provider = RasterData(WorldClim2, BioClim)
opts = (; resolution=2.5)
temperature = SimpleSDMPredictor(provider, layer=1; opts..., boundingbox...)
precipitation = SimpleSDMPredictor(provider, layer=8; opts..., boundingbox...)

# Presence layer
presence_layer = similar(temperature, Bool)
for i in axes(occ, 1)
    if ~isnothing(presence_layer[occ[i]...])
        presence_layer[occ[i]...] = true
    end
end

# Background
possible_background = pseudoabsencemask(DistanceToEvent, presence_layer) * cellsize(temperature)

# Absence layer
absence_layer = backgroundpoints(
    (x -> x^1.01).(possible_background),
    3sum(presence_layer);
    replace=false
)

# Cleanup
replace!(absence_layer, false => nothing)
replace!(absence_layer, true => false)
replace!(presence_layer, false => nothing)

y = vcat(
    rename(DataFrame(presence_layer), :value => :presence),
    rename(DataFrame(absence_layer), :value => :presence)
)

# Get the full data
dfs = [rename(DataFrame(SimpleSDMPredictor(provider, layer=i; opts..., boundingbox...)), :value => layers(provider)[i]) for i in 1:19]
Xf = dropmissing(reduce((x,y) -> leftjoin(x,y; on=[:latitude, :longitude]), dfs))

# Get the training data
Xy = dropmissing(leftjoin(y, Xf, on=[:longitude, :latitude]))