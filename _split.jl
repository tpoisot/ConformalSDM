using DataFrames
import CSV
using MLJ
using Statistics

Xy = DataFrame(CSV.File("training.csv"))
Xy.presence = coerce(Xy.presence, OrderedFactor)

Xtrain, Xtest = partition(Xy, 0.7, shuffle=true)

vars = Symbol.("BIO" .* string.([19, 10, 4, 14, 16]))
coordinates = select(Xtrain, [:latitude, :longitude])
KMeans = @load KMeans pkg=Clustering

function makefolds(folds)
    return [(findall(folds .!= i), findall(folds .== i)) for i in sort(unique(folds))]
end

y, X = unpack(select(Xtrain, Not([:longitude, :latitude])), ==(:presence))
X = X[:,vars]

ty, tX = unpack(select(Xtest, Not([:longitude, :latitude])), ==(:presence))
tX = tX[:,vars]

# Split space / environment
kmach = machine(KMeans(k=10), coordinates) |> fit!
spatialfold = predict(kmach, coordinates)
kmach = machine(KMeans(k=10), X) |> fit!
envirofold = predict(kmach, Xtrain)

Tree = @load EvoTreeClassifier pkg = EvoTrees
tree = Tree(nbins=128, max_depth=6)

mach = machine(tree, X, y)
fit!(mach)

performance_measures = [f1score, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, balanced_accuracy, matthews_correlation, accuracy]

# On the testing holdout
pr = predict_mode(mach, tX)
for pm in performance_measures
    @info pm, pm(pr, ty)
end

# Validation values
evaluate(tree, X, y,
    resampling=CV(nfolds=10, shuffle=true; rng=12345),
    measures=performance_measures,
    verbosity=0
)

evaluate(tree, X, y,
    resampling=StratifiedCV(nfolds=10, shuffle=true; rng=12345),
    measures=performance_measures,
    verbosity=0
)

evaluate(tree, X, y,
    resampling=makefolds(spatialfold),
    measures=performance_measures,
    verbosity=0
)

evaluate(tree, X, y,
    resampling=makefolds(envirofold),
    measures=performance_measures,
    verbosity=0
)

