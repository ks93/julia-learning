Pkg.activate("julia-play")

using Plots
using RDatasets
using LinearAlgebra

function predict_lr(X::Array, β::Array) :: Array
    return X*β
end

function fit_lr(X::Array, y::Array) :: Array
    # β_hat = (X^T*X)^-1 * X^T * y
    Xt = transpose(X)
    return inv(Xt*X)*Xt*y
end


mtcars = dataset("datasets", "mtcars")

x  = convert(Array, mtcars.HP)
X = cat(ones(length(x)), x, x.^2, dims=2)

y = convert(Array, mtcars.MPG)


gr()
scatter(x,y, xlabel="HP", ylabel="MPG", labels="observation")

β_hat = fit_lr(X, y)

x_ = range(minimum(x)-10, maximum(x)+10, length=1000)
X_ = cat(ones(length(x_)), x_, x_.^2, dims=2)
y_hat = predict_lr(X_, β_hat)

plot!(x_, y_hat, labels="pred")
