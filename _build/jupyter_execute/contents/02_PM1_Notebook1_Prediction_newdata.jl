using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Dates")
Pkg.add("Plots")
using CSV
using DataFrames
using Dates
using Plots

#Reading the CSV file into a DataFrame
#We have to set the category type for some variable
data = CSV.File("wage2015_subsample_inference.csv"; types = Dict("occ" => String,"occ2"=> String,"ind"=>String,"ind2"=>String)) |> DataFrame
println("Number of Rows : ", size(data)[1],"\n","Number of Columns : ", size(data)[2],) #rows


[eltype(col) for col = eachcol(data)]

# first 10 lines of the data
first(data,10)

#a quick decribe of the data
describe(data)

n = size(data)[1]
z = select(data, Not([:rownames, :lwage, :wage]))
p = size(z)[2] 

println("Number of observations : ", n, "\n","Number of raw regressors:", p )

z_subset = select(data, ["lwage","sex","shs","hsg","scl","clg","ad","mw","so","we","ne","exp1"])
describe(z_subset, :mean)

#Needed Packages and extra just in case
Pkg.add("Plots")
Pkg.add("Lathe")
Pkg.add("GLM")
Pkg.add("StatsPlots")
Pkg.add("MLBase")

# Load the installed packages
using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase

#basic model
basic  = @formula(lwage ~ (sex + exp1 + shs + hsg+ scl + clg + mw + so + we + occ2+ ind2))
basic_results  = lm(basic, data)

#flexible model
flex = @formula(lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we))
regflex = lm(flex, data)

Pkg.add("Lasso")
using Lasso

flex = @formula(lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we))
lasso_model = fit(LassoModel, flex, data)
