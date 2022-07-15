# import Pkg; Pkg.add("RData")
# import Pkg; Pkg.add("CodecBzip2")
# import Pkg; Pkg.add("DataStructures")
# import Pkg; Pkg.add("NamedArrays")
# import Pkg; Pkg.add("PrettyTables")
# import Pkg; Pkg.add("Lasso")
# import Pkg; Pkg.add("Libz")
# import Pkg; Pkg.add("PlotlyJS")
# import Pkg; Pkg.add("StatsBase")

using StatsBase

using RData, LinearAlgebra, GLM, DataFrames, Statistics, Random, Distributions, DataStructures, NamedArrays, PrettyTables
import CodecBzip2

using CSV
using DataFrames



data = CSV.File( "../data/wage2015_subsample_inference.csv"; types = Dict("occ" => String,"occ2"=> String,"ind"=>String,"ind2"=>String)) |> DataFrame

size( data )

Z = select( data, Not( ["lwage", "wage"]) )
names( Z )

using StatsPlots
using Distributions

histogram(rand(Normal(), 1000))

using Plots

plt = Plots.histogram( data[!, "wage"], 
    title = "Empirical wage distribution from the US survey data", 
    nbins = 35, 
    label = "")
xlabel!( "g(X)" )
ylabel!( "hourly wage" )
display( plt )

using StatsBase
plot1 = fit(Histogram, data[!, "wage"], nbins=35 )
collect( plot1.edges[1] )

Random.seed!(1234)

training = sample( collect(1:nrow( data ) ), trunc(Int, 3 * nrow( data ) / 4 ),  replace= false )

data_train = data[ vec(training), : ]
data_test = data[ Not(training), : ]

size(data_test)

size(data_train)

X_basic =  "sex + exp1 + exp2+ shs + hsg+ scl + clg + mw + so + we + occ2+ ind2"
X_flex = "sex + exp1 + exp2 + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)"

map(*, "lwage", "~", X_basic )

using GLM

formula_basic = @formula(lwage ~ (sex + exp1 + exp2+ shs + hsg+ scl + clg + mw + so + we + occ2+ ind2) )
formula_flex = @formula(lwage ~ (sex + exp1 + exp2 + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)) )

model_X_basic_train = ModelMatrix(ModelFrame(formula_basic,data_train)).m
model_X_basic_test = ModelMatrix(ModelFrame(formula_basic,data_test)).m
p_basic = size(model_X_basic_test)[2]

model_X_flex_train = ModelMatrix(ModelFrame(formula_flex,data_train)).m
model_X_flex_test = ModelMatrix(ModelFrame(formula_flex,data_test)).m
p_flex = size(model_X_flex_test)[2]

Y_train = data_train[!, "lwage"]
Y_test = data_test[ !,  "lwage"]

p_basic
p_flex

data_train

size( data_train )

fit_lm_basic = lm(formula_basic, data_train)

using DataFrames, GLM

# Compute the Out-Of-Sample Performance
yhat_lm_basic = GLM.predict( fit_lm_basic , data_test )
res_lm_basic = ( Y_test - yhat_lm_basic ) .^ 2
print("The mean squared error (MSE) using the basic model is equal to " , mean( res_lm_basic ) ) # MSE OLS (basic model)    

matrix_ones = ones( size(res_lm_basic)[1] ,1 )
mean_residuals = lm(  matrix_ones, res_lm_basic )
MSE_lm_basic = [ coef( mean_residuals ) , stderror( mean_residuals ) ]
MSE_lm_basic

R2_lm_basic = 1 .- ( MSE_lm_basic[1] / var( Y_test ) )
print( "The R^2 using the basic model is equal to ", R2_lm_basic ) # MSE OLS (basic model) 

# ols (flexible model)
fit_lm_flex = lm( formula_flex, data_train ) 
yhat_lm_flex = GLM.predict( fit_lm_flex, data_test)

res_lm_flex = ( Y_test - yhat_lm_flex ) .^ 2
mean_residuals = lm(  matrix_ones, res_lm_flex )
MSE_lm_flex = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

R2_lm_flex = 1 .- ( MSE_lm_flex[1] / var( Y_test ) )
print( "The R^2 using the basic model is equal to ", R2_lm_basic ) # MSE OLS (flex model) 

# lasso and versions
# library(hdm) 
# fit_rlasso  = rlasso(formula_basic, data_train, post= FALSE)
# fit_rlasso_post = rlasso(formula_basic, data_train, post= TRUE)

yhat_rlasso   = predict(fit_rlasso, newdata= data_test)
yhat_rlasso_post   = predict(fit_rlasso_post, newdata= data_test)

res_rlasso = ( Y_test - yhat_rlasso ) .^ 2
mean_residuals = lm(  matrix_ones, res_rlasso )
MSE_rlasso = [ coef( mean_residuals ) , stderror( mean_residuals ) ]


res_rlasso_post = ( Y_test - yhat_rlasso_post ) .^ 2
mean_residuals = lm(  matrix_ones, res_rlasso_post )
MSE_rlasso_post = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

R2_rlasso = 1 .- ( MSE_rlasso[1] / var( Y_test ) )
R2_rlasso_post = 1 .- ( MSE_rlasso_post[1] / var( Y_test ) )

print( "The R^2 using the basic model is equal to ", R2_rlasso, " for lasso and ", R2_rlasso_post," for post-lasso") # R^2 lasso/post-lasso (basic model) 

# fit_rlasso_flex  = rlasso( formula_flex, data_train, post=FALSE )
# fit_rlasso_post_flex = rlasso( formula_flex, data_train, post=TRUE )

yhat_rlasso_flex   = predict( fit_rlasso_flex, newdata=data_test )
yhat_rlasso_post_flex   = predict( fit_rlasso_post_flex, newdata=data_test )

res_rlasso_flex = ( Y_test - yhat_rlasso_flex ) .^ 2
mean_residuals = lm(  matrix_ones, res_rlasso_flex )
MSE_rlasso_flex = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

res_rlasso_post_flex = ( Y_test - yhat_rlasso_post_flex ) .^ 2
mean_residuals = lm(  matrix_ones, res_rlasso_post_flex )
MSE_rlasso_post_flex = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

R2_rlasso_flex = 1 .- ( MSE_rlasso_flex[1] / var( Y_test ) )
R2_rlasso_post_flex = 1 .- ( MSE_rlasso_post_flex[1] / var( Y_test ) )

print( "The R^2 using the flexible model is equal to ", R2_lasso_flex, " for lasso and ", 
    R2_lasso_post_flex, " for post-lasso" ) # R^2 lasso/post-lasso ( flexible model ) 

using GLMNet

fit_lasso_cv   = GLMNet.glmnetcv(model_X_basic_train, Y_train, alpha=1)
fit_ridge   = GLMNet.glmnetcv(model_X_basic_train, Y_train, alpha=0)
fit_elnet   = GLMNet.glmnetcv(model_X_basic_train, Y_train, alpha= 0.5)

yhat_lasso_cv    = GLMNet.predict(fit_lasso_cv,  model_X_basic_test)
yhat_ridge   = GLMNet.predict(fit_ridge,  model_X_basic_test)
yhat_elnet   = GLMNet.predict(fit_elnet,  model_X_basic_test)

res_lasso_cv = ( Y_test - yhat_lasso_cv ) .^ 2
mean_residuals = lm(  matrix_ones, res_lasso_cv )
MSE_lasso_cv = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

res_ridge = ( Y_test - yhat_ridge ) .^ 2
mean_residuals = lm(  matrix_ones, res_ridge )
MSE_ridge = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

res_elnet = ( Y_test - yhat_elnet ) .^ 2
mean_residuals = lm(  matrix_ones, res_elnet )
MSE_elnet = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

R2_lasso_cv = 1 .- ( MSE_lasso_cv[1] / var( Y_test ) )
R2_ridge = 1 .- ( MSE_ridge[1] / var( Y_test ) )
R2_elnet = 1 .- ( MSE_elnet[1] / var( Y_test ) )

print("R^2 using cross-validation for lasso, ridge and elastic net in the basic model:",R2_lasso_cv,R2_ridge,R2_elnet)

fit_lasso_cv_flex   = GLMNet.glmnetcv(model_X_flex_train, Y_train, alpha=1)
fit_ridge_flex   = GLMNet.glmnetcv(model_X_flex_train, Y_train, alpha=0)
fit_elnet_flex   = GLMNet.glmnetcv(model_X_flex_train, Y_train, alpha= 0.5)

yhat_lasso_cv_flex    = GLMNet.predict(fit_lasso_cv_flex,  model_X_flex_test)
yhat_ridge_flex   = GLMNet.predict(fit_ridge_flex,  model_X_flex_test)
yhat_elnet_flex   = GLMNet.predict(fit_elnet_flex,  model_X_flex_test)

res_lasso_cv_flex = ( Y_test - yhat_lasso_cv_flex ) .^ 2
mean_residuals = lm(  matrix_ones, res_lasso_cv_flex )
MSE_lasso_cv_flex = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

res_ridge_flex = ( Y_test - yhat_ridge_flex ) .^ 2
mean_residuals = lm(  matrix_ones, res_ridge_flex )
MSE_ridge_flex = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

res_elnet_flex = ( Y_test - yhat_elnet_flex ) .^ 2
mean_residuals = lm(  matrix_ones, res_elnet_flex )
MSE_elnet_flex = [ coef( mean_residuals ) , stderror( mean_residuals ) ]

R2_lasso_cv_flex = ( 1 .- ( MSE_lasso_cv_flex[1] / var( Y_test ) ) )[1]
R2_ridge_flex = ( 1 .- ( MSE_ridge_flex[1] / var( Y_test ) ) )[1]
R2_elnet_flex = ( 1 .- ( MSE_elnet_flex[1] / var( Y_test ) ) )[1]

# import Pkg; Pkg.add( "MLJ" )

# import Pkg; Pkg.add( "MLJModels" )

# import Pkg; Pkg.add( "DecisionTree" )
import Pkg; Pkg.add( "ScikitLearn" )

using MLJ # using the MLJ framework
using MLJModels # loads the modesl

using ScikitLearn, Random

@sk_import tree: DecisionTreeRegressor

trees = DecisionTreeRegressor( random_state = 0, min_impurity_decrease = 0.001 )

ScikitLearn.fit!( trees, model_X_basic_train, Y_train )

ScikitLearn.cost_complexity_pruning_path( trees )

reshape( Y_train, 1 )

using DecisionTree

fit_trees = DecisionTree.build_tree( labels = Y_train, features = model_X_basic_train, min_purity_increase = 0.001)



# R^2 using cross-validation (flexible model) 
print( "R^2 using cross-validation for lasso, ridge and elastic net in the flexible model:",R2_lasso_cv_flex,R2_ridge_flex,R2_elnet_flex)

R2_lasso_cv_flex[1]

# ols (basic model)
lm_basic = sm.OLS( Y_train, model_X_basic_train )
fit_lm_basic = lm_basic.fit()

# Compute the Out-Of-Sample Performance
yhat_lm_basic = fit_lm_basic.predict( model_X_basic_test )
print( f"The mean squared error (MSE) using the basic model is equal to , {np.mean((Y_test-yhat_lm_basic)**2)} ") # MSE OLS (basic model)    

resid_basic = (Y_test-yhat_lm_basic)**2

MSE_lm_basic = sm.OLS( resid_basic , np.ones( resid_basic.shape[0] ) ).fit().summary2().tables[1].iloc[0, 0:2]
MSE_lm_basic

R2_lm_basic = 1 - ( MSE_lm_basic[0]/Y_test.var() )
print( f"The R^2 using the basic model is equal to, {R2_lm_basic}" ) # MSE OLS (basic model) 

# ols (flex model)
lm_flex = sm.OLS( Y_train, model_X_flex_train )
fit_lm_flex = lm_flex.fit()

yhat_lm_flex = fit_lm_flex.predict( model_X_flex_test )

resid_flex = (Y_test-yhat_lm_flex)**2

MSE_lm_flex = sm.OLS( resid_flex , np.ones( resid_flex.shape[0] ) ).fit().summary2().tables[1].iloc[0, 0:2]
MSE_lm_flex

R2_lm_flex = 1 - ( MSE_lm_flex[0]/Y_test.var() )
print( f"The R^2 using the flex model is equal to, {R2_lm_flex}" ) # MSE OLS (flex model) 

import hdmpy

fit_rlasso = hdmpy.rlasso( model_X_basic_train.to_numpy() , Y_train.to_numpy().reshape( Y_train.size , 1 ) , post = False )
fit_rlasso_post = hdmpy.rlasso( model_X_basic_train.to_numpy() , Y_train.to_numpy().reshape( Y_train.size , 1 ) , post = True )

# Getting mean of each variable
meanx = model_X_basic_test.mean( axis = 0 ).values.\
                        reshape( model_X_basic_test.shape[ 1 ] , 1 )

# Reducing the mean
new_x1 = model_X_basic_test.to_numpy() - \
                    (np.ones( ( model_X_basic_test.shape[ 0 ] , 1 ) ) @ meanx.T)

# Getting the significant variables
x1_est_rlasso = new_x1[ :, fit_rlasso.est['index'].iloc[:, 0].to_list()]

# Getting the coef. from significant variables
beta_rlasso = fit_rlasso.est['beta'].loc[ fit_rlasso.est['index'].\
                                     iloc[:, 0].to_list(), ].to_numpy()

# yhat
yhat_rlasso = (x1_est_rlasso @ beta_rlasso) + np.mean( Y_test.to_numpy() )
residuals_rlasso = Y_test.to_numpy().reshape( Y_test.to_numpy().size, 1)  - yhat_rlasso

# Getting mean of each variable
meanx = model_X_basic_test.mean( axis = 0 ).values.\
                        reshape( model_X_basic_test.shape[ 1 ] , 1 )

# Reducing the mean
new_x1 = model_X_basic_test.to_numpy() - \
                    (np.ones( ( model_X_basic_test.shape[ 0 ] , 1 ) ) @ meanx.T)

# Getting the significant variables
x1_est_rlasso_post = new_x1[ :, fit_rlasso_post.est['index'].iloc[:, 0].to_list()]

# Getting the coef. from significant variables
beta_rlasso_post = fit_rlasso_post.est['beta'].loc[ fit_rlasso_post.est['index'].\
                                     iloc[:, 0].to_list(), ].to_numpy()

# yhat
yhat_rlasso_post = (x1_est_rlasso_post @ beta_rlasso_post) + np.mean( Y_test.to_numpy() )
residuals_rlasso_post = Y_test.to_numpy().reshape( Y_test.to_numpy().size, 1)  - yhat_rlasso_post

MSE_lasso = sm.OLS( ( residuals_rlasso )**2 , np.ones( yhat_rlasso.size )  ).fit().summary2().tables[1].round(3)
MSE_lasso_post = sm.OLS( ( residuals_rlasso_post )**2  , np.ones( yhat_rlasso_post.size )  ).fit().summary2().tables[1].round(3)

R2_lasso = 1 - MSE_lasso.iloc[0, 0]/ np.var( Y_test )
R2_lasso_post = 1 - MSE_lasso_post.iloc[0, 0]/ np.var( Y_test )

print( f"The R^2 using the basic model is equal to {R2_lasso},for lasso and {R2_lasso_post} for post-lasso") # R^2 lasso/post-lasso (basic model) 

fit_rlasso_flex = hdmpy.rlasso( model_X_flex_train.to_numpy() , Y_train.to_numpy().reshape( Y_train.size , 1 ) , post = False )
fit_rlasso_post_flex = hdmpy.rlasso( model_X_flex_train.to_numpy() , Y_train.to_numpy().reshape( Y_train.size , 1 ) , post = True )

# Getting mean of each variable
meanx = model_X_flex_test.mean( axis = 0 ).values.\
                        reshape( model_X_flex_test.shape[ 1 ] , 1 )

# Reducing the mean
new_x1 = model_X_flex_test.to_numpy() - \
                    (np.ones( ( model_X_flex_test.shape[ 0 ] , 1 ) ) @ meanx.T)

# Getting the significant variables
x1_est_rlasso_flex = new_x1[ :, fit_rlasso_flex.est['index'].iloc[:, 0].to_list()]

# Getting the coef. from significant variables
beta_rlasso_flex = fit_rlasso_flex.est['beta'].loc[ fit_rlasso_flex.est['index'].\
                                     iloc[:, 0].to_list(), ].to_numpy()

# yhat
yhat_rlasso_flex = (x1_est_rlasso_flex @ beta_rlasso_flex) + np.mean( Y_test.to_numpy() )
residuals_rlasso_flex = Y_test.to_numpy().reshape( Y_test.to_numpy().size, 1)  - yhat_rlasso_flex

# Getting mean of each variable
meanx = model_X_flex_test.mean( axis = 0 ).values.\
                        reshape( model_X_flex_test.shape[ 1 ] , 1 )

# Reducing the mean
new_x1 = model_X_flex_test.to_numpy() - \
                    (np.ones( ( model_X_flex_test.shape[ 0 ] , 1 ) ) @ meanx.T)

# Getting the significant variables
x1_est_rlasso_post_flex = new_x1[ :, fit_rlasso_post_flex.est['index'].iloc[:, 0].to_list()]

# Getting the coef. from significant variables
beta_rlasso_post_flex = fit_rlasso_post_flex.est['beta'].loc[ fit_rlasso_post_flex.est['index'].\
                                     iloc[:, 0].to_list(), ].to_numpy()

# yhat
yhat_rlasso_post_flex = (x1_est_rlasso_post_flex @ beta_rlasso_post_flex) + np.mean( Y_test.to_numpy() )
residuals_rlasso_post_flex = Y_test.to_numpy().reshape( Y_test.to_numpy().size, 1)  - yhat_rlasso_post_flex

MSE_lasso_flex = sm.OLS( ( residuals_rlasso_flex )**2 , np.ones( yhat_rlasso_flex.size )  ).fit().summary2().tables[1].round(3)
MSE_lasso_post_flex = sm.OLS( ( residuals_rlasso_post_flex )**2  , np.ones( yhat_rlasso_post_flex.size )  ).fit().summary2().tables[1].round(3)

R2_lasso_flex = 1 - MSE_lasso.iloc[0, 0]/ np.var( Y_test )
R2_lasso_post_flex = 1 - MSE_lasso_post_flex.iloc[0, 0]/ np.var( Y_test )

print( f"The R^2 using the basic model is equal to {R2_lasso_flex},for lasso and {R2_lasso_post_flex} for post-lasso") # R^2 lasso/post-lasso (basic model) 

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
import statsmodels.api as sm

# Reshaping Y variable
Y_vec = Y_train.to_numpy().reshape( Y_train.to_numpy().size, 1)

# Scalar distribution
scaler = StandardScaler()
scaler.fit( Y_vec )
std_Y = scaler.transform( Y_vec )

# Regressions
fit_lasso_cv_basic = LassoCV(cv = 10 , random_state = 0 , normalize = True ).fit( model_X_basic_train, std_Y )
fit_ridge_basic = ElasticNetCV( cv = 10 , normalize = True , random_state = 0 , l1_ratio = 0.0001 ).fit( model_X_basic_train , std_Y )
fit_elnet_basic = ElasticNetCV( cv = 10 , normalize = True , random_state = 0 , l1_ratio = 0.5, max_iter = 100000 ).fit( model_X_basic_train , std_Y )

# Predictions
yhat_lasso_cv_basic = scaler.inverse_transform( fit_lasso_cv_basic.predict( model_X_basic_test ) )
yhat_ridge_basic = scaler.inverse_transform( fit_ridge_basic.predict( model_X_basic_test ) )
yhat_elnet_basic = scaler.inverse_transform( fit_elnet_basic.predict( model_X_basic_test ) )

MSE_lasso_cv_basic = sm.OLS( ((Y_test - yhat_lasso_cv_basic)**2 ) , np.ones( yhat_lasso_cv_basic.shape )  ).fit().summary2().tables[1].round(3)
MSE_ridge_basic = sm.OLS( ((Y_test - yhat_ridge_basic)**2 ) , np.ones( yhat_ridge_basic.size )  ).fit().summary2().tables[1].round(3)
MSE_elnet_basic = sm.OLS( ((Y_test - yhat_elnet_basic)**2 ) , np.ones( yhat_elnet_basic.size )  ).fit().summary2().tables[1].round(3)
# our coefficient of MSE_elnet are far from r output

R2_lasso_cv_basic = 1- MSE_ridge_basic.iloc[0,0] / np.var( Y_test )
R2_ridge_basic = 1- MSE_lasso_cv_basic.iloc[0,0] / np.var( Y_test )
R2_elnet_basic = 1- MSE_elnet_basic.iloc[0,0] / np.var( Y_test )

print( f"R^2 using cross-validation for lasso, ridge and elastic net in the basic model: {R2_lasso_cv_basic},{R2_ridge_basic},{R2_elnet_basic}")

# Reshaping Y variable
Y_vec = Y_train.to_numpy().reshape( Y_train.to_numpy().size, 1)

# Scalar distribution
scaler = StandardScaler()
scaler.fit( Y_vec )
std_Y = scaler.transform( Y_vec )

# Regressions
fit_lasso_cv_flex = LassoCV(cv = 10 , random_state = 0 , normalize = True ).fit( model_X_flex_train, std_Y )
fit_ridge_flex = ElasticNetCV( cv = 10 , normalize = True , random_state = 0 , l1_ratio = 0.0001 ).fit( model_X_flex_train , std_Y )
fit_elnet_flex = ElasticNetCV( cv = 10 , normalize = True , random_state = 0 , l1_ratio = 0.5, max_iter = 100000 ).fit( model_X_flex_train , std_Y )

# Predictions
yhat_lasso_cv_flex = scaler.inverse_transform( fit_lasso_cv_flex.predict( model_X_flex_test ) )
yhat_ridge_flex = scaler.inverse_transform( fit_ridge_flex.predict( model_X_flex_test ) )
yhat_elnet_flex = scaler.inverse_transform( fit_elnet_flex.predict( model_X_flex_test ) )

MSE_lasso_cv_flex = sm.OLS( ((Y_test - yhat_lasso_cv_flex)**2 ) , np.ones( yhat_lasso_cv_flex.shape )  ).fit().summary2().tables[1].round(3)
MSE_ridge_flex = sm.OLS( ((Y_test - yhat_ridge_flex)**2 ) , np.ones( yhat_ridge_flex.size )  ).fit().summary2().tables[1].round(3)
MSE_elnet_flex = sm.OLS( ((Y_test - yhat_elnet_flex)**2 ) , np.ones( yhat_elnet_flex.size )  ).fit().summary2().tables[1].round(3)
# our coefficient of MSE_elnet are far from r output

R2_lasso_cv_flex = 1- MSE_ridge_flex.iloc[0,0] / np.var( Y_test )
R2_ridge_flex = 1- MSE_lasso_cv_flex.iloc[0,0] / np.var( Y_test )
R2_elnet_flex = 1- MSE_elnet_flex.iloc[0,0] / np.var( Y_test )

print( f"R^2 using cross-validation for lasso, ridge and elastic net in the basic model: {R2_lasso_cv_flex},{R2_ridge_flex},{R2_elnet_flex}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from scipy.sparse import diags
from IPython.display import Image, display

trees = DecisionTreeRegressor( random_state = 0, min_impurity_decrease = 0.001 )

pd.DataFrame(trees.cost_complexity_pruning_path( y_basic_train, model_X_basic_train ))

trees_fit =  trees.fit( y_basic_train, model_X_basic_train )

# tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree( trees_fit , filled = True , rounded = True  )

dir(trees_fit)















table= np.zeros( (15, 3) )
table[0,0:2]   = MSE_lm_basic
table[1,0:2]   = MSE_lm_flex
table[2,0:2]   = MSE_lasso.iloc[0, [0, 1]]
table[3,0:2]   = MSE_lasso_post.iloc[0, [0, 1]]
table[4,0:2]   = MSE_lasso_flex.iloc[0, [0, 1]]
table[5,0:2]   = MSE_lasso_post_flex.iloc[0, [0, 1]]
table[6,0:2]   = MSE_lasso_cv_basic.iloc[0, [0, 1]]
table[7,0:2]   = MSE_ridge_basic.iloc[0, [0, 1]]
table[8,0:2]   = MSE_elnet_basic.iloc[0, [0, 1]]
table[9,0:2]   = MSE_lasso_cv_flex.iloc[0, [0, 1]]
table[10,0:2]  = MSE_ridge_flex.iloc[0, [0, 1]]
table[11,0:2]  = MSE_elnet_flex.iloc[0, [0, 1]]
# table[13,1:2]  = MSE_rf
# table[14,1:2]  = MSE_boost
# table[15,1:2]  = MSE_pt



table[0,2]   = R2_lm_basic
table[1,2]   = R2_lm_flex
table[2,2]   = R2_lasso
table[3,2]   = R2_lasso_post
table[4,2]   = R2_lasso_flex
table[5,2]   = R2_lasso_post_flex
table[6,2]   = R2_lasso_cv_basic
table[7,2]   = R2_ridge_basic
table[8,2]   = R2_elnet_basic
table[9,2]   = R2_lasso_cv_flex
table[10,2]  = R2_ridge_flex
table[11,2]  = R2_elnet_flex
# table[13,3]  = R2_rf
# table[14,3]  = R2_boost
# table[15,3]  = R2_pt




colnames_table= ["MSE", "S_E_ for MSE", "R-squared"]
rownames_table= ["Least Squares (basic)","Least Squares (flexible)", "Lasso", "Post-Lasso","Lasso (flexible)","Post-Lasso (flexible)", \
                    "Cross-Validated lasso", "Cross-Validated ridge","Cross-Validated elnet","Cross-Validated lasso (flexible)","Cross-Validated ridge (flexible)","Cross-Validated elnet (flexible)",  \
                    "Random Forest","Boosted Trees", "Pruned Tree"]
table_pandas = pd.DataFrame( table, columns = colnames_table )
table_pandas.index = rownames_table

table_pandas = table_pandas.round(3)
table_html = table_pandas.to_latex()
table_pandas


