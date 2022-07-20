# import Pkg; Pkg.add("Flux")

using RData, LinearAlgebra, GLM, DataFrames, Statistics, Random, Distributions, DataStructures, NamedArrays, PrettyTables, Plots
import CodecBzip2

# Importing .Rdata file
rdata_read = load("../data/wage2015_subsample_inference.RData")

# Since rdata_read is a dictionary, we check if there is a key called "data", the one we need for our analyze
haskey(rdata_read, "data")

# Now we save that dataframe with a new name
data = rdata_read["data"]
names(data)

typeof(data), size(data)

Z =  select(data, ["lwage", "wage"])     # regressors

Random.seed!(1234) 
training = sample(1:nrow(data), Int(floor(nrow(data)*(3/4))), replace = false)

data_train = data[training,1:16]
data_test = data[Not(training),1:16]
data_train

size(data_train), size(data_test)

# normalize the data

mean_1 = mean.(eachcol(data_train))
mean_1 = [names(data_train) mean_1]

std_1 = std.(eachcol(data_train))
std_1 = [names(data_train) std_1]

df = DataFrame()
for i in 1:size(data_train)[2]
     p = (data_train[!, i] .- mean_1[i,2]) / std_1[i,2]
     colname = names(data_train)[i]
     df[!,colname] = p
end
data_train = df
data_train

df = DataFrame()
for i in 1:size(data_test)[2]
     p = (data_test[!, i] .- mean_1[i,2]) / std_1[i,2]
     colname = names(data_test)[i]
     df[!,colname] = p
end
data_test = df
data_test

typeof(data_train), typeof(data_test)

formula_basic = @formula(lwage ~ sex + exp1 + shs + hsg+ scl + clg + mw + so + we)
println(formula_basic)

model_X_basic_train = ModelMatrix(ModelFrame(formula_basic, data_train)).m
model_X_basic_test = ModelMatrix(ModelFrame(formula_basic, data_test)).m

Y_train = data_train[!,"lwage"]
Y_test = data_test[!,"lwage"]

# df = DataFrame()
# for i in 1:size(data_train)[2]
#      p = normalize(data_train[!,i])
#      colname = names(data_train)[i]
#      df[!,colname] = p
# end
# df
# # data_test = df
# # data_test


# normalize(data_train[!,])

size(Y_train)

model_X_basic_train

size(model_X_basic_train)

using Flux
using Flux: crossentropy, @epochs
using Flux.Data: DataLoader
using Flux: throttle
using Flux: onehotbatch, onecold, @epochs

model_nn = Chain(Dense(size(model_X_basic_train,2), 20, relu), 
              Dense(20, size(model_X_basic_train,2), relu),
              Dense(10, 1))

# compile the Flux model
model = model_nn
#opt(x, y) = Descent()
opt1 = ADAM()
loss(x, y) = Flux.Losses.mse(model(x), y)
metrics(x, y) = Flux.mae(model(x), y)

ps = Flux.params(model)

model

# m1 = zeros(size(model_X_basic_train, 1), size(model_X_basic_train, 2) - 1 )
# Y_train = hcat(Y_train, m1)

loss(model_X_basic_train', Y_train')

opt(model_X_basic_train', Y_train)

data = DataLoader((model_X_basic_train', Y_train'))

# training the network 

#@epochs 1000 Flux.train!(loss, ps, data, opt, cb = throttle(() -> println("training"), 10))
@epochs 1000 Flux.train!(loss, ps, data, opt1, cb = throttle(() -> println("training"), 10))

# # evaluating the performnace
# model %>% evaluate(model_X_basic_test, Y_test, verbose = 0)

# Calculating the performance measures 

pred_nn = model(model_X_basic_test')'
# We change the type of this variable in order to make numeric operations
pred_nn = vec(pred_nn)

resid_basic = (Y_test .- pred_nn).^2

one = vec(ones(size(Y_test,1),1))

resid_basic = DataFrame([resid_basic], [:resid_basic])

one = DataFrame([one], [:one])
data_aux = [resid_basic one]

fm = @formula(resid_basic ~ 0 + one)

MSE_nn_basic = lm(fm, data_aux, dropcollinear=false)

@show coef(MSE_nn_basic)
@show stderror(MSE_nn_basic)

R2_nn_basic = 1 .- (coef(MSE_nn_basic) / var(Y_test))
println("The R^2 using NN is equal to ", R2_nn_basic) # MSE NN (basic model) 

plot(Y_test, label = "Y_test")
plot!(pred_nn, label = "pred_nn")
plot!(size=(950,600))
