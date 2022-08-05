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

# to_install = ["CSV", "DataFrames", "Dates", "Plots"]
# using Pkg 
# Pkg.add(to_install)
# PKG.add("Lathe")
# Pkg.add("HTTP")
using CSV, DataFrames, Dates, Plots, Lathe, GLM, Statistics, MLBase, HTTP

#Reading the CSV file into a DataFrame
#We have to set the category type for some variable
url = "https://github.com/d2cml-ai/14.388_jl/raw/main/data/wage2015_subsample_inference.csv"
data = CSV.File(download(url); types = Dict("occ" => String,"occ2"=> String,"ind"=>String,"ind2"=>String)) |> DataFrame
size(data)

#a quick decribe of the data
describe(data)

n = size(data)[1]
z = select(data, Not([:rownames, :lwage, :wage]))
p = size(z)[2] 

println("Number of observations : ", n, "\n","Number of raw regressors: ", p )


z_subset = select(data, ["lwage","sex","shs","hsg","scl","clg","ad","mw","so","we","ne","exp1"])
rename!(z_subset, ["Log Wage", "Sex", "Some High School", "High School Graduate", "Some College", "College Graduate", "Advanced Degree", "Midwest", "South", "West", "Northeast", "Experience"])

describe(z_subset, :mean)

#basic model
basic  = @formula(lwage ~ (sex + exp1 + shs + hsg+ scl + clg + mw + so + we + occ2+ ind2))
basic_results  = lm(basic, data)
println(basic_results)
println("Number of regressors in the basic model: ", size(coef(basic_results), 1))

#flexible model
flex = @formula(lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we))
flex_results = lm(flex, data)
println(flex_results)
println("Number of regressors in the flexible model: ", size(coef(flex_results), 1))

using Lasso

lasso_model = fit(LassoModel, flex, data)

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
    
    return p, r_2, adj_r2, mse, mse_adj
end

p1, r2_1, r2_adj1, mse1, mse_adj1 = ref_bsc(basic_results);

p2, r2_2, r2_adj2, mse2, mse_adj2 = ref_bsc(flex_results);

pL, r2_L, r2_adjL, mseL, mse_adjL = ref_bsc(lasso_model, true);

println("R-squared for the basic model: ", r2_1)
println("Adjusted R-squared for the basic model: ", r2_adj1)
println("R-squared for the flexible model: ", r2_2)
println("Adjusted R-squared for the flexible model: ", r2_adj2)
println("R-squared for the lasso with flexible model: ", r2_2)
println("Adjusted R-squared for the lasso with flexible model: ", r2_adj2, "\n")

println("MSE for the basic model: ", mse1)
println("MSE for the basic model: ", mse_adj1)
println("MSE for the flexible model: ", mse2)
println("MSE for the flexible model: ", mse_adj2)
println("MSE for the lasso with flexible model: ", mseL)
println("MSE for the lasso with flexible model: ", mse_adjL)


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
    Flexible_reg = ref_bsc(flex_results),
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
