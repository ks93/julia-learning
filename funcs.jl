# Pkg.activate("julia-play")

using CSV
using HTTP
using DataFrames
using Plots
using StatsPlots

using ScikitLearn



# We want a language that’s open source, with a liberal license.
# We want the speed of C with the dynamism of Ruby.
# We want a language that’s homoiconic, with true macros like Lisp,
# but with obvious, familiar mathematical notation like Matlab.
# We want something as usable for general programming as Python,
# as easy for statistics as R, as natural for string processing as Perl,
# as powerful for linear algebra as Matlab,
# as good at gluing programs together as the shell.
# Something that is dirt simple to learn,
# yet keeps the most serious hackers happy.
# We want it interactive and we want it compiled.
#
# (Did we mention it should be as fast as C?)

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
col_symbols = Symbol.(col_names)

f = CSV.File(HTTP.get(data_url).body, header=false, ignoreemptylines=true)
df_iris = DataFrame(f)
rename!(df_iris, col_symbols)

plotlyjs()
default(size=(1000,600))
scatter(df_iris.sepal_length, df_iris.sepal_width, group=df_iris.class)

feature_names = col_symbols[(1:4)]
label_name = col_symbols[5]

train_mask = rand(nrow(df_iris)) .< 0.8

df_train = df_iris[train_mask,:]
df_test = df_iris[.!train_mask,:]

X_train = convert(Array, df_train[:, feature_names])
y_train = convert(Array, df_train[:, label_name])

X_test = convert(Array, df_test[:, feature_names])
y_test = convert(Array, df_test[:, label_name])

# Real ML
@sk_import linear_model: LogisticRegression

lr = LogisticRegression(fit_intercept=true)

fit!(lr, X_train, y_train)

train_accuracy = sum(predict(lr, X_train) .== y_train) / length(y_train)
println("train accuracy: $train_accuracy")

test_accuracy = sum(predict(lr, X_test) .== y_test) / length(y_test)
println("test accuracy: $test_accuracy")
