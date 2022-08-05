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

# If necesary, install functions
# import Pkg; Pkg.add("GLM")
# import Pkg; Pkg.add("DataFrames")

# Import functions
using LinearAlgebra, GLM, DataFrames, Statistics, Random

Random.seed!(1234)

n = 1000
p = n

# Create a 1000x1000 matrix of standard Gaussians
X = randn(n, p)

# Create a 1000x1 matrix of standard Gaussians
Y = randn(n)

# We can not run the regression below, because we need to have n>p otherwise error shows up.(I think it is because the matrix
# decomposition procedure)
# Fitted linear regression 
# fitted = lm(X,Y)

# This is a fuction that returns coeficients,R2 and Adj R2

function OLSestimator(Y, X)

    β = inv(X'*X)*(X'*Y)
    # β = X\Y
    errors = Y - X*β
    R_squared = 1.0 - sum(errors.^2.0)/sum((Y .- mean(Y)).^2.0)
    R_squared_adj =  1.0 - ( 1.0 - R_squared )*( size(Y)[1] - 1.0 )/( size(Y)[1]- size(X)[2] - 1.0 )    
    
    return β, R_squared, R_squared_adj
end

results_ols = OLSestimator(Y, X)

println("p/n is")
println(p/n)

println("R2 is")
println(results_ols[2])

println("Adjusted R2 is")
println(results_ols[3])

# We have to make sure that both variables are the same type (Integers or floats) to avoid errors when running the regression
n = 1000;
p = Int(n/2);

# Create a nxp matrix of standard Gaussians
X = randn(n, p);

# Create a nx1 matrix of standard Gaussians
Y = randn(n);

fitted = lm(X,Y);

println("p/n is")
println(p/n)

println("R2 is")
println(r2(fitted))

println("Adjusted R2 is")
println(adjr2(fitted))

n = 1000
p = Int(0.05*n)

X = randn(n, p)

Y = randn(n)

fitted = lm(X,Y)

println("p/n is")
println(p/n)

println("R2 is")
println(r2(fitted))

println("Adjusted R2 is")
println(adjr2(fitted))
