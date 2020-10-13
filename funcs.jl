# Pkg.activate("julia-play")

using Plots
using CSV
using HTTP
using DataFrames
using StatsPlots

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
col_symbols = Symbol.(col_names)#[Symbol(cn) for cn in col_names]

f = CSV.File(HTTP.get(data_url).body, header=false, ignoreemptylines=true)
iris = DataFrame(f)
rename!(iris, col_symbols)

plotly()
scatter(iris.sepal_length, iris.sepal_width, group=iris.class)
