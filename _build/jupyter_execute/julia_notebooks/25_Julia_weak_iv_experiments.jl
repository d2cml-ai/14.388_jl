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

# Import relevant packages
# Import relevant packages
# using Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("GLM")
# Pkg.add("FixedEffectModels")
# Pkg.add("DecisionTree")
# Pkg.add("PrettyTables")
# Pkg.add("CovarianceMatrices")
# Pkg.add("RegressionTables")
# Pkg.add("StatsFuns")
# Pkg.add("Plots")
# Pkg.add("RData")
# Pkg.add("MLBase")

using CSV, DataFrames, FixedEffectModels, DecisionTree, PrettyTables, CovarianceMatrices, RegressionTables, StatsFuns, Plots, RData, MLBase, GLM

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


