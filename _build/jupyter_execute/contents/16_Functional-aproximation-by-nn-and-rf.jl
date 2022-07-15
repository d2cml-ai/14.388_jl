#import packages
using Pkg
Pkg.add("Distributions")
Pkg.add("DecisionTree") 
Pkg.add("Plots")
using LinearAlgebra, DataFrames, Statistics, Random, Distributions,DecisionTree, Plots

rand(1)
X_train = rand(1000,1)
f1 = x->exp(4*x)
Y_train = vec(f1.(X_train));

TreeModel = build_tree(Y_train, X_train)
# apply learned model
predTM = apply_tree(TreeModel, X_train)


scatter(X_train, Y_train, type="p", pch=19, xlab="z", ylab="g(z)",alpha=.8)
plot!(X_train, predTM, lt = :scatter )


Pkg.add("DecisionTree")
Pkg.add("ScikitLearn")
Pkg.add("PyPlot")
using DecisionTree, ScikitLearn, PyPlot

reg= DecisionTreeRegressor(pruning_purity_threshold = 0.1)

#using DecisionTree: fit as fit
DecisionTree.fit!(reg, X_train,Y_train)

predic1 = DecisionTree.predict(reg,X_train);

scatter(X_train, Y_train, type="p", pch=19, xlab="z", ylab="g(z)",alpha=.8)
scatter!(X_train,predic1   )

tree = DecisionTreeRegressor(pruning_purity_threshold = 0.5) #
DecisionTree.fit!(tree, X_train,Y_train)
pred2 = DecisionTree.predict(tree,X_train)
scatter(X_train, Y_train, type="p", pch=19, xlab="z", ylab="g(z)",alpha=.8, label = "actual")
scatter!(X_train,pred2 ,label = "pred")

RFmodel = build_forest(Y_train, X_train)
pred_RF = apply_forest(RFmodel,X_train)
scatter(X_train, Y_train, type="p", pch=19, xlab="z", ylab="g(z)", label = "real data", title = "Random Forest")
scatter!(X_train,pred_RF, label= "RF pred")

Pkg.add("JLBoost"),Pkg.add("StatsBase"),Pkg.add("MLJBase"),Pkg.add("MLJ")
Pkg.add(url ="https://github.com/Evovest/EvoTrees.jl") 
using StatsBase: sample
using EvoTrees,MLJBase, MLJ,JLBoost,EvoTrees

tree_model = EvoTreeRegressor(loss=:linear, max_depth=4, η=0.01, nrounds=100)

mach = machine(tree_model, X_train, Y_train)

MLJ.fit!(mach)

yhat = MLJ.predict(mach, X_train);

scatter(X_train, Y_train,type="p", pch=19, xlab="z", ylab="g(z)", label = "real data", title = "Boosted Tree")
scatter!(X_train, yhat, label = "boosted tree pred")

Pkg.add("XGBoost")
using  XGBoost
using XGBoost: predict as predict

bst = xgboost(X_train, 100, label = Y_train, eta = 0.01, max_depth = 4);

pred_xg = XGBoost.predict(bst, X_train);

scatter(X_train, Y_train,type="p", pch=19, xlab="z", ylab="g(z)", label = "real data", title = "Boosted Tree")
scatter!(X_train, pred_xg, label = "boosted tree pred")

using Pkg
Pkg.add("Flux")
Pkg.add("CUDA")
using Flux ,CUDA

#building the model
layer1 = Dense(200, 200,relu)
layer2 = Dense(20, 20, relu  )
layer3 = Dense(1,1)
model = Chain( 
    layer1,
    layer2,
    layer3)


#building the predict function
predict = Dense(1000, 1000)

pred1 =  predict(X_train);
loss(x, y) = Flux.Losses.mse(predict(x), y)
loss(X_train, Y_train)
scatter(X_train, Y_train, label = "real data")
scatter!(X_train, pred1, label = "Pred 1 epoch")

loss(x, y) = Flux.Losses.mse(predict(x), y)
println("The MSE for this first prediction without optimizer is ", loss(X_train, Y_train))

#we add a optimizer and join the data
opt = Descent()
data = [(X_train, Y_train)]

predict.weight;
predict.bias;

parameters = Flux.params(predict)

predict.weight in parameters, predict.bias in parameters

#training the model
using Flux: train!
train!(loss, parameters, data, opt)

#the loss function change
println("The MSE for this second model with optimizer is ", loss(X_train, Y_train))
pred2 = predict(X_train)
scatter(X_train, Y_train, label = "real data")
scatter!(X_train, pred2, label = "pred 1 epochs")

#running 100 epochs
for epoch in 1:100
    train!(loss, parameters, data, opt)
  end

print("The MSE with 100 epochs is ",loss(X_train, Y_train))
pred100 = predict(X_train)
scatter(X_train, Y_train, label = "real data")
scatter!(X_train, pred100, label = "pred 100 epochs")

