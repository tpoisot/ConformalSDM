include("_getdata.jl")

using MLJ

# Split the data to train a model
schema(Xy)
Xy.presence = coerce(Xy.presence, OrderedFactor)

y, X = unpack(select(Xy, Not([:longitude, :latitude])), ==(:presence); rng=420)

models(matching(X,y))

Tree = @load EvoTreeClassifier pkg=EvoTrees
tree = Tree()

evaluate(tree, X, y,
                resampling=CV(shuffle=true),
                        measures=[matthews_correlation, accuracy],
                        verbosity=0)

