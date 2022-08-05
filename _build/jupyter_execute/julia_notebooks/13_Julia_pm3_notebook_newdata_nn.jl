# !wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
# !dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
# !apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
# !apt update -q
# !apt install cuda gcc-6 g++-6 -y -q
# !ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
# !ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++

# !curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz" -o julia.tar.gz
# !tar -xzf julia.tar.gz -C /usr --strip-components 1
# !rm -rf julia.tar.gz*
# !julia -e 'using Pkg; pkg"add IJulia; precompile"'

# import Pkg; Pkg.add("Flux")

using RData, LinearAlgebra, GLM, DataFrames, Statistics, Random, Distributions, DataStructures, NamedArrays, PrettyTables, Plots
import CodecBzip2

# Importing .Rdata file
url = "https://github.com/d2cml-ai/14.388_jl/raw/github_data/data/wage2015_subsample_inference.RData"
download(url, "data.RData")
rdata_read = load("data.RData")
rm("data.RData")
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
