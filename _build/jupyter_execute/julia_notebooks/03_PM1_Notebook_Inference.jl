# using Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Dates")
# Pkg.add("Plots")
# Pkg.add("GLMNet")
using GLMNet
using CSV
using DataFrames
using Dates
using Plots
using Statistics,RData

rdata_read = load("../data/wage2015_subsample_inference.RData")
data = rdata_read["data"]
names(data)
println("Number of Rows : ", size(data)[1],"\n","Number of Columns : ", size(data)[2],) #rows and columns

Z = select(data, ["lwage","sex","shs","hsg","scl","clg","ad","ne","mw","so","we","exp1"])

data_female = filter(row -> row.sex == 1, data)
Z_female = select(data_female,["lwage","sex","shs","hsg","scl","clg","ad","ne","mw","so","we","exp1"] )

data_male = filter(row -> row.sex == 0, data)
Z_male = select(data_male,["lwage","sex","shs","hsg","scl","clg","ad","ne","mw","so","we","exp1"] )

means = DataFrame( variables = names(Z), All = describe(Z, :mean)[!,2], Men = describe(Z_male,:mean)[!,2], Female = describe(Z_female,:mean)[!,2])


mean(Z_female[:,:lwage]) - mean(Z_male[:,:lwage])

#install all the package that we can need
# Pkg.add("Plots")
# Pkg.add("Lathe")
# Pkg.add("GLM")
# Pkg.add("StatsPlots")
# Pkg.add("MLBase")
# Pkg.add("Tables")

# Load the installed packages
using DataFrames
using CSV
using Tables
using Plots
using Lathe
using GLM



nocontrol_model = lm(@formula(lwage ~ sex),data)
nocontrol_est = GLM.coef(nocontrol_model)[2]
nocontrol_se = GLM.coeftable(nocontrol_model).cols[2][2]

println("The estimated gender coefficient is ", nocontrol_est ," and the corresponding robust standard error is " ,nocontrol_se )

flex = @formula(lwage ~ sex + (exp1+exp2+exp3+exp4) * (shs+hsg+scl+clg+occ2+ind2+mw+so+we))
control_model = lm(flex , data)
control_est = GLM.coef(control_model)[2]
control_se = GLM.coeftable(control_model).cols[2][2]
println(control_model)
println("Coefficient for OLS with controls " , control_est)

# models
# model for Y
flex_y = @formula(lwage ~ (exp1+exp2+exp3+exp4) * (shs+hsg+scl+clg+occ2+ind2+mw+so+we))
flex_d = @formula(sex ~ (exp1+exp2+exp3+exp4) * (shs+hsg+scl+clg+occ2+ind2+mw+so+we))

# partialling-out the linear effect of W from Y
t_Y = residuals(lm(flex_y, data))

# partialling-out the linear effect of W from D
t_D = residuals(lm(flex_d, data))

data_res = DataFrame(t_Y = t_Y, t_D = t_D )
# regression of Y on D after partialling-out the effect of W
partial_fit = lm(@formula(t_Y ~ t_D), data_res)
partial_est = GLM.coef(partial_fit)[2]

println("Coefficient for D via partiallig-out ", partial_est)

# standard error
partial_se = GLM.coeftable(partial_fit).cols[2][2]

#condifence interval
GLM.confint(partial_fit)[2,:]

# models
# model for Y
flex_y = @formula(lwage ~  (exp1+exp2+exp3+exp4) * (shs+hsg+scl+clg+occ2+ind2+mw+so+we));

# model for D
flex_d = @formula(sex ~ (exp1+exp2+exp3+exp4) * (shs+hsg+scl+clg+occ2+ind2+mw+so+we));

# Pkg.add("Lasso")
using Lasso

lasso_y = fit(LassoModel, flex_y, data,  ?? = 0.1)
t_y = residuals(lasso_y)

lasso_d = fit(LassoModel, flex_d, data, ?? = 0.1)
t_d = residuals(lasso_d)

data_res = DataFrame(t_Y = t_y, t_D = t_d )

partial_lasso_fit = lm(@formula(t_Y ~ t_D), data_res)
partial_lasso_est = GLM.coef(partial_lasso_fit)[2]
partial_lasso_se = GLM.coeftable(partial_lasso_fit).cols[2][2]

println("Coefficient for D via partialling-out using lasso ", partial_lasso_est)

DataFrame(modelos = [ "Without controls", "full reg", "partial reg", "partial reg via lasso" ], 
Estimate = [nocontrol_est,control_est,partial_est, partial_lasso_est], 
StdError = [nocontrol_se,control_se, partial_se, partial_lasso_se])

# import Pkg
# Pkg.add("StatsModels")
# Pkg.add("Combinatorics")
# Pkg.add("IterTools")
# we have to configure the package internaly with the itertools package, this because 
#julia dont iunderstand (a formula) ^2, it takes as an entire term not as interactions 
#between variables

#this code fix the problem mencioned above
using StatsModels, Combinatorics, IterTools

combinations_upto(x, n) = Iterators.flatten(combinations(x, i) for i in 1:n)
expand_exp(args, deg::ConstantTerm) =
    tuple(((&)(terms...) for terms in combinations_upto(args, deg.n))...)

StatsModels.apply_schema(t::FunctionTerm{typeof(^)}, sch::StatsModels.Schema, ctx::Type) =
    apply_schema.(expand_exp(t.args_parsed...), Ref(sch), ctx)

StatsModels.apply_schema(t::FunctionTerm{typeof(^)}, sch::StatsModels.FullRank, ctx::Type) =
    apply_schema.(expand_exp(t.args_parsed...), Ref(sch), ctx)

extra_flex = @formula(lwage ~  sex + (exp1+exp2+exp3+exp4+shs+hsg+scl+clg+occ2+ind2+mw+so+we)^2)

control_fit = lm(extra_flex, data)
control_est = GLM.coef(control_fit)[2]

println("Number of Extra-Flex Controls: ", size(modelmatrix(control_fit))[2] -1) #minus the intercept
println("Coefficient for OLS with extra flex controls ", control_est)

#std error
control_se = GLM.stderror(control_fit)[2];

extraflex_y = @formula(lwage ~ (exp1+exp2+exp3+exp4+shs+hsg+scl+clg+occ2+ind2+mw+so+we)^2)# model for Y
extraflex_d = @formula(sex ~ (exp1+exp2+exp3+exp4+shs+hsg+scl+clg+occ2+ind2+mw+so+we)^2) # model for D

# partialling-out the linear effect of W from Y
t_y = residuals(fit(LassoModel, extraflex_y, data,standardize = false))
# partialling-out the linear effect of W from D
t_d = residuals(fit(LassoModel, extraflex_d, data,standardize = false))

data_partial = DataFrame(t_y = t_y, t_d = t_d )

# regression of Y on D after partialling-out the effect of W
partial_lasso_fit = lm(@formula(t_y~t_d), data_partial)

partial_lasso_est = GLM.coef(partial_lasso_fit)[2]

println("Coefficient for D via partialling-out using lasso :", partial_lasso_est)

#standard error

partial_lasso_se = GLM.stderror(partial_lasso_fit)[2];


tabla3 = DataFrame(modelos = [ "Full reg", "partial reg via lasso" ], 
Estimate = [control_est,partial_lasso_est], 
StdError = [control_se,partial_lasso_se])
