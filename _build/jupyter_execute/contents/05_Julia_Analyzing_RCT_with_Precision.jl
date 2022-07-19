# import Pkg; Pkg.add("Distributions")
# import Pkg; Pkg.add("Tables")
# import Pkg; Pkg.add("TableOperations")
# import Pkg; Pkg.add("StatsBase")
# import Pkg; Pkg.add("FreqTables")
# import Pkg; Pkg.add("Plots")

# Import relevant packages for splitting data
using LinearAlgebra, GLM, DataFrames, Statistics, Random, Distributions, Tables, TableOperations, StatsBase, FreqTables, DataFrames

# Set Seed
# to make the results replicable (generating random numbers)
Random.seed!(12345676)     # set MC seed

n = 1000                # sample size
Z = randn(n, 1)         # generate Z
Y0 = -Z + randn(n, 1)   # conditional average baseline response is -Z
Y1 = Z + randn(n, 1)    # conditional average treatment effect is +Z
D = Int.(rand(Uniform(), n, 1) .< 0.2)   # treatment indicator; only 23% get treated
length(D[D .== 1])*100/length(D[D .== 0])  # treatment indicator; only 23% get treated
mean(D)

Y = (Y1.*D) + (Y0.*(ones(n,1)-D))    # observed Y
D = D - fill(mean(D),n,1)            # demean D
Z = Z - fill(mean(Z),n,1)            # demean Z

Z_times_D = Z.*D
X = hcat(D, Z, Z_times_D)
data = DataFrame(X, [:Z, :D, :Z_times_D])

# Import packages for OLS regression
using GLM, Plots

data_aux = [Y D Z Z_times_D]
data_aux = DataFrame(data_aux, [:Y, :D, :Z, :Z_times_D])

fm_1 = @formula(Y ~ D)
fm_2 = @formula(Y ~ D + Z)
fm_3 = @formula(Y ~ D + Z + Z_times_D)

CL_model = lm(fm_1, data_aux)
CRA_model = lm(fm_2, data_aux)  #classical
IRA_model = lm(fm_3, data_aux)  #interactive approach
# Standard deviations for estimators
CL = sqrt(sum((Y - predict(CL_model)).*(Y - predict(CL_model)))./length(Y))
CRA = sqrt(sum((Y - predict(CRA_model)).*(Y - predict(CRA_model)))./length(Y))
IRA = sqrt(sum((Y - predict(IRA_model)).*(Y - predict(IRA_model)))./length(Y))
@show CL
@show CRA
@show IRA

# Check t values of regressors 
@show coeftable(CL_model).cols[4]
@show coeftable(CRA_model).cols[4]
@show coeftable(IRA_model).cols[4]

# Comparing models
ftest(CL_model.model, CRA_model.model, IRA_model.model)

@show CL_model
@show CRA_model
@show IRA_model

Random.seed!(12345676)     # set MC seed
n = 1000
B = 1000

# format of data = float32
CLs = fill(0., B)
CRAs = fill(0., B)
IRAs = fill(0., B)


# formulas for regressions
fm_1 = @formula(Y ~ D)
fm_2 = @formula(Y ~ D + Z)
fm_3 = @formula(Y ~ D + Z + Z_times_D)

# models
CL_model = lm(fm_1, data_aux)
CRA_model = lm(fm_2, data_aux)  #classical
IRA_model = lm(fm_3, data_aux)  #interactive approach


# simulation
for i in 1:B
    Z = randn(n, 1)         # generate Z
    Y0 = -Z + randn(n, 1)   # conditional average baseline response is -Z
    Y1 = Z + randn(n, 1)    # conditional average treatment effect is +Z
    D = Int.(rand(Uniform(), n, 1) .< 0.2)   # treatment indicator; only 23% get treated

    Y = (Y1.*D) + (Y0.*(ones(n,1)-D))    # observed Y

    D = D - fill(mean(D),n,1)            # demean D
    Z = Z - fill(mean(Z),n,1)            # demean Z

    Z_times_D = Z.*D
    X = hcat(D, Z, Z_times_D)
    data_aux = [Y D Z Z_times_D]
    data_aux = DataFrame(data_aux, [:Y, :D, :Z, :Z_times_D])
    
    CLs[i,] = predict(CL_model)[i]
    CRAs[i,] = predict(CRA_model)[i]
    IRAs[i,] = predict(IRA_model)[i]

end


# check  standard deviations
println("Standard deviations for estimators")  
println("CL model: " , sqrt(sum((Y - predict(CL_model)).*(Y - predict(CL_model)))./length(Y)))
println("CRA model: " , sqrt(sum((Y - predict(CL_model)).*(Y - predict(CRA_model)))./length(Y)))
println("IRA model: " , sqrt(sum((Y - predict(CL_model)).*(Y - predict(IRA_model)))./length(Y)))
