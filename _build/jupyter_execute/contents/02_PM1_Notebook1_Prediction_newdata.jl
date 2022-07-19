# to_install = ["CSV", "DataFrames", "Dates", "Plots"]
# using Pkg 
# Pkg.add(to_install)

using CSV
using DataFrames
using Dates
using Plots, Lathe

#Reading the CSV file into a DataFrame
#We have to set the category type for some variable
data = CSV.File("../data/wage2015_subsample_inference.csv"; types = Dict("occ" => String,"occ2"=> String,"ind"=>String,"ind2"=>String)) |> DataFrame
println("Number of Rows : ", size(data)[1],"\n","Number of Columns : ", size(data)[2],) #rows

# first 10 lines of the data
first(data,5)

#a quick decribe of the data
describe(data)

n = size(data)[1]
z = select(data, Not([:rownames, :lwage, :wage]))
p = size(z)[2] 

println("Number of observations : ", n, "\n","Number of raw regressors: ", p )

z_subset = select(data, ["lwage","sex","shs","hsg","scl","clg","ad","mw","so","we","ne","exp1"])
describe(z_subset, :mean)

#Needed Packages and extra just in case
# to_install = ["Plots", "Lathe", "GLM", "StatPlots", "MLBase"]
# using Pkg; Pkg.add(to_install)

# Load the installed packages
using DataFrames
using CSV
using Plots
using GLM
using Statistics
# using StatsPlots
using MLBase

#basic model
basic  = @formula(lwage ~ (sex + exp1 + shs + hsg+ scl + clg + mw + so + we + occ2+ ind2))
basic_results  = lm(basic, data)

#flexible model
flex = @formula(lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we))
regflex = lm(flex, data)

# using Pkg
# Pkg.add("Lasso")
using Lasso

flex = @formula(lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we))
lasso_model = fit(LassoModel, flex, data)

# lasso_model, basic_results, regflex
n_data = size(data)[1]

function ref_bsc(model, lasso = false, n = n_data)
     if lasso
        p = length(coef(model))
        y_hat = predict(model)
        y_r = data.lwage
        r_2 = 1 - sum((y_r .- y_hat).^2)  / sum((y_r .- mean(y_r)).^2)
        adj_r2 = 1 - (1 - r_2) * ((n - 1) / (n - p - 1))
    else
        p = length(coef(model))
        r_2 = r2(model)
        adj_r2 = adjr2(model)
    end   
    
    mse = mean(residuals(model).^2)
    mse_adj = (n / (n - p)) * mse
    
    ref = [p, r_2, adj_r2, mse, mse_adj]
    
    return ref
    
end

DataFrame(
    Model = ["p", "R^2", "MSE", "R^2 adjusted", "MSE adjusted"],
    Basic_reg = ref_bsc(basic_results),
    Flexxibmle_reg = ref_bsc(regflex),
    lasso_flex = ref_bsc(lasso_model, true)
)

using Lathe.preprocess: TrainTestSplit

train, test = TrainTestSplit(data, 4/5)
reg_basic = lm(basic, train)

train_reg_basic = predict(reg_basic, test)
y_test = test.lwage

mse_test1 = sum((y_test .- train_reg_basic).^2) / length(y_test)
r2_test1 = 1 - mse_test1 / var(y_test)

print("Test MSE for the basic model: $mse_test1\nTest R2 for the basic model: $r2_test1")

reg_flex = lm(flex, train)
train_reg_flex = predict(reg_flex, test)
mse_test2 = sum((y_test .- train_reg_flex).^2) / length(y_test)
r2_test2 = 1 - mse_test2 / var(y_test)
    
print("Test MSE for the basic model: $mse_test2\nTest R2 for the basic model: $r2_test2")

reg_lasso = fit(LassoModel, flex, train)
train_reg_lasso = predict(reg_lasso, test)
mse_lasso = sum((y_test .- train_reg_lasso).^2) / length(y_test)
r2_lasso = 1 - mse_lasso / var(y_test)
print("Test MSE for the basic model: $mse_lasso\nTest R2 for the basic model: $r2_lasso")

MSE = [mse_test1, mse_test2, mse_lasso]
R2 = [r2_test1, r2_test2, r2_lasso]
Model = ["Basic reg", "Flexible reg", "Lasso Regression"]
DataFrame( Model = Model, MSE_test = MSE, R2_test = R2)
