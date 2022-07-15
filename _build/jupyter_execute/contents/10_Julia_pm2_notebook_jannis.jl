# import Pkg; Pkg.add("RData")
# import Pkg; Pkg.add("CodecBzip2")
# import Pkg; Pkg.add("DataStructures")
# import Pkg; Pkg.add("NamedArrays")
# import Pkg; Pkg.add("PrettyTables")
# import Pkg; Pkg.add("Lasso")

using RData, LinearAlgebra, GLM, DataFrames, Statistics, Random, Distributions, DataStructures, NamedArrays, PrettyTables
import CodecBzip2

# Importing .Rdata file
growth_read = load("../data/GrowthData.RData")

# Since growth_read is a dictionary, we check if there is a key called "GrowthData", the one we need for our analyze
haskey(growth_read, "GrowthData")

# Now we save that dataframe with a new name
growth = growth_read["GrowthData"]
names(growth)

typeof(growth), size(growth)

y = growth[!, "Outcome"]
y = DataFrame([y], [:y])

X = select(growth, Not(["Outcome"]))

size(X)

typeof(y), typeof(X)

data = [y X]

names(data)

# OLS regression
reg_ols  = lm(term(:Outcome) ~ sum(term.(names(growth[!, Not(["Outcome", "intercept"])]))), growth, dropcollinear=false)

# OLS regression
reg_ols  = lm(term(:y) ~ sum(term.(names(data[!, Not(["y", "intercept"])]))), data, dropcollinear=false)

# output: estimated regression coefficient corresponding to the target regressor
est_ols = coef(reg_ols)[2]

# output: std. error
std_ols = stderror(reg_ols)[2]

# output: 95% confidence interval
lower_ci = coeftable(reg_ols).cols[5][2]
upper_ci = coeftable(reg_ols).cols[6][2]

table_1 = NamedArray(zeros(1, 4))
setnames!(table_1, ["OLS"], 1)
table_1

table_1[1] = est_ols
table_1[2] = std_ols   
table_1[3] = lower_ci
table_1[4] = upper_ci    
 
table_1

table_1_pandas = DataFrame( table_1, [ :"Estimator", :"Std. Error", :"lower bound CI", :"upper bound CI"])
model_1 = DataFrame(model = ["OLS"])
table_1_pandas = hcat(model_1,table_1_pandas)
table_1_pandas

header = (["Model", "Estimator", "Std. Error", "lower bound CI", "upper bound CI"])
pretty_table(table_1_pandas; backend = Val(:html), header = header, formatters=ft_round(6), alignment=:c)

# Importing .Rdata file
growth_read = load("../data/GrowthData.RData")
# Now we save that dataframe with a new name
growth = growth_read["GrowthData"]

# Create main variables
Y = growth[!, "Outcome"]
Y = DataFrame([Y], [:Y])
D = growth[!, "gdpsh465"]
D = DataFrame([D], [:D])
W = select(growth, Not(["Outcome", "intercept", "gdpsh465"]))
data = [Y D W]

using Lasso, GLM

# Seat values for Lasso

lasso_model = fit(LassoModel, term(:Y) ~  sum(term.(names(data[!, Not(["Y", "D"])]))), data; α = 0.8)
r_Y = residuals(lasso_model)
r_Y = DataFrame([r_Y], [:r_Y])

# Part. out d

lasso_model = fit(LassoModel, term(:D) ~  sum(term.(names(data[!, Not(["Y", "D"])]))), data;  α = 0.8)
r_D = residuals(lasso_model)
r_D = DataFrame([r_D], [:r_D])

# ols 
data_aux = [r_Y r_D]
fm_1 = @formula(r_Y ~ r_D)
partial_lasso_fit = lm(fm_1, data_aux)

# output: estimated regression coefficient corresponding to the target regressor
est_lasso = coef(partial_lasso_fit)[2]

# output: std. error
std_lasso = stderror(partial_lasso_fit)[2]

# output: 95% confidence interval
lower_ci_lasso = coeftable(partial_lasso_fit).cols[5][2]
upper_ci_lasso = coeftable(partial_lasso_fit).cols[6][2]

# Regress residuales
partial_lasso_fit = lm(fm_1, data_aux)
partial_lasso_est = coef(partial_lasso_fit)[2]

println("Coefficient for D via partialling-out using lasso is: ", partial_lasso_est )

table_2 = NamedArray(zeros(1, 4))
setnames!(table_2, ["DOUBLE LASSO"], 1)
table_2

table_2[1] = est_lasso
table_2[2] = std_lasso   
table_2[3] = lower_ci_lasso
table_2[4] = upper_ci_lasso    
 
table_2

table_2_pandas = DataFrame( table_2, [ :"Estimator", :"Std. Error", :"lower bound CI", :"upper bound CI"])
model_2 = DataFrame(model = ["DOUBLE LASSO"])
table_2_pandas = hcat(model_2,table_2_pandas)
table_2_pandas

header = (["Model", "Estimator", "Std. Error", "lower bound CI", "upper bound CI"])
pretty_table(table_2_pandas; backend = Val(:html), header = header, formatters=ft_round(6), alignment=:c)

table_3 = vcat(table_1, table_2)
setnames!(table_3, ["OLS", "DOUBLE LASSO"], 1)
table_3

table_3_pandas = DataFrame( table_3, [ :"Estimator", :"Std. Error", :"lower bound CI", :"upper bound CI"])
model_3 = DataFrame(model = ["OLS", "DOUBLE LASSO"])
table_3_pandas = hcat(model_3, table_3_pandas)
table_3_pandas

header = (["Model", "Estimator", "Std. Error", "lower bound CI", "upper bound CI"])
pretty_table(table_3_pandas; backend = Val(:html), header = header, formatters=ft_round(6), alignment=:c)

using Plots

xerror = table_3_pandas[!,2] .- table_3_pandas[!,4]

scatter(table_3_pandas[!,2], ["OLS", "Double Lasso"], label = "Estimator", xerrors = xerror, 
        xtick = -1:0.008:1, linestyle = :dash, seriescolor=:orange)
plot!(size=(950,400))

include("hdmjl/hdmjl.jl")

res_Y_0 = rlasso_arg( W, Y, nothing, true, true, true, false, false, 
                    nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )

res_Y = rlasso(res_Y_0)

res_Y = res_Y["residuals"]

res_D_0 = rlasso_arg( W, D, nothing, true, true, true, false, false, 
                    nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )
res_D = rlasso(res_D_0)["residuals"]
# We need to convert the vector into matrix because the lm function requires "X" to be matrix 
res_D = reshape(res_D, length(res_D), 1)

lm(res_D, res_Y)




