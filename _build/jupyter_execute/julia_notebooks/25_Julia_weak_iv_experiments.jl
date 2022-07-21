# Import relevant packages
using Pkg
Pkg.add("CSV"), using CSV
Pkg.add("DataFrames"), using DataFrames
Pkg.add("GLM"), using GLM
Pkg.add("FixedEffectModels"), using FixedEffectModels
Pkg.add("PrettyTables"), using PrettyTables
Pkg.add("CovarianceMatrices"), using CovarianceMatrices
Pkg.add("RegressionTables"), using RegressionTables
Pkg.add("StatsFuns"), using StatsFuns
Pkg.add("Plots"), using Plots
Pkg.add("MLBase"), using MLBase
Pkg.add("Seaborn"), using Seaborn
Pkg.add("Random"), using Random
Pkg.add("Statistics"), using Statistics

function foo1(a;rng=MersenneTwister(3))
  return randn(rng,a)
end

function foo2(a;rng=MersenneTwister(1))
  return randn(rng,a)
end

    B = 1000
    IVEst = zeros( B )
    n = 100
    beta = 0.25

    U = foo1(n)
    Z = foo2(n)
    D = beta*Z+U
    Y = D + U;
    intercept = ones(length(U))
    data1 = DataFrame(intercept = intercept, U = U, Z = Z, D = D, Y = Y);

    mod = reg(data1, @formula(D ~ Z))

IV =  reg(data1, @formula(Y ~ 0 + (D ~ Z)))
IV

IV.coef

# dependent variable ~ exogenous variables + (endogenous variables ~ instrumental variables)

# Set seed
B = 1000
IVEst = zeros(B)


for i in 1:B
    
    U = randn( n)
    Z = randn( n)
    D = beta*Z+U
    Y = D + U
    intercept = ones(length(U))
    data2 = DataFrame(intercept = intercept, U = U, Z = Z, D = D, Y = Y);
        
    IV =  reg(data2, @formula(Y ~ + (D ~  Z)))
    
    IVEst[i,1] = IV.coef[2]
end

println(minimum(IVEst))
println(maximum(IVEst))

IVEst

val = collect(range( -5, 5.5, step = 0.05 ))
var = (1/beta^2)*(1/100) # theoretical variance of IV
sd = sqrt(var)

μ=0; σ=sd
d = Normal(μ, σ)
normal_dist = rand(d,1000)

# plotting both distibutions on the same figure
Seaborn.kdeplot(x = IVEst.-1, shade = true, color = "red")
Seaborn.kdeplot(x = normal_dist, shade = true, color = "blue")
Seaborn.title("Actual Distribution vs Gaussian")
Seaborn.xlabel("IV Estimator -True Effect")
Seaborn.xlim(-5,5)


