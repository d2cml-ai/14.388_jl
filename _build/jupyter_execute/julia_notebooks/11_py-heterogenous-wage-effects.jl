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

using RData, LinearAlgebra, GLM, DataFrames, Statistics, Random, Distributions, DataStructures, NamedArrays, PrettyTables, StatsModels, Combinatorics, Plots

import CodecBzip2

# Importing .Rdata file
url = "https://github.com/d2cml-ai/14.388_jl/raw/github_data/data/cps2012.RData"
download(url, "cps2012.RData")
cps2012 = load("cps2012.RData")
rm("cps2012.RData")
cps2012 = cps2012["data"]
names(cps2012)

    # couples variables combinations 
    combinations_upto(x, n) = Iterators.flatten(combinations(x, i) for i in 1:n)

    # combinations without same couple
    expand_exp(args, deg::ConstantTerm) =
        tuple(((&)(terms...) for terms in combinations_upto(args, deg.n))...)

    StatsModels.apply_schema(t::FunctionTerm{typeof(^)}, sch::StatsModels.Schema, ctx::Type) =
        apply_schema.(expand_exp(t.args_parsed...), Ref(sch), ctx)

# Basic model 


reg = @formula(lnw ~ -1 + female + female&(widowed + divorced + separated + nevermarried +
hsd08 + hsd911 + hsg + cg + ad + mw + so + we + exp1 + exp2 + exp3) + (widowed +
divorced + separated + nevermarried + hsd08 + hsd911 + hsg + cg + ad + mw + so +
we + exp1 + exp2 + exp3)^2 )


formula_basic = apply_schema(reg, schema(reg, cps2012))


formula_basic

# get variabl'es name

coefnames(formula_basic)

Y = select(cps2012,:lnw)  # uptcome variable
control = coefnames(formula_basic)[2]  # regresors control 
names_col = Symbol.(control)  # string to Symbol to create varaible's name 

X = StatsModels.modelmatrix(formula_basic,cps2012)

X = DataFrame(X, names_col)

# Function to get index of constant columns   

cons_column = []


for i in 1:size(X,2)
    if var(X[!,i]) == 0
        append!(cons_column  , i)      
    end       
end


# Drop constant columns 

names(X)[cons_column]
select!(X, Not(names(X)[cons_column]))

# demean function
function desv_mean(a)
    a = Matrix(a)   # dataframe to matrix 
    A = mean(a, dims = 1)
    M = zeros(Float64, size(X,1), size(X,2))
    
    for i in 1:size(a,2)
          M[:,i] = a[:,i] .- A[i]
    end
    
    return M
end    


# Matrix Model & demean

X = DataFrame(desv_mean(X), names(X)) # Dataframe and names 

# index to get columns that contains female

index = []

for i in 1:size(X,2)  
        if contains( names(X)[i] , "female")
            append!(index, i)
        end  
end

index

# Control variables 

W = select(X, Not(names(X)[index]))

# load HDM package

include("hdmjl/hdmjl.jl")

for i in index
    print(names(X)[i], "\n")
end

table = NamedArray(zeros(16, 2))

j = 0

X_resid = Vector{Float64}()  # empty vector - float 

for i in index

j = j + 1
    
#first step
D = select(X, names(X)[i])
    
D_reg_0  = rlasso_arg( W, D, nothing, true, true, true, false, false, 
                    nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )

D_resid =  rlasso(D_reg_0)["residuals"]

    
append!( X_resid, D_resid )

    
end 

W

#second step

Y_reg_0  = rlasso_arg( W, Y, nothing, true, true, true, false, false, 
                    nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )

Y_resid = rlasso(Y_reg_0)["residuals"]

X_resid = reshape(X_resid, length(Y_resid), length(index))  # reshape stacked vector to mattrix 


# third step
    
Lasso_HDM = lm(X_resid, Y_resid)

GLM.coeftable(Lasso_HDM).cols[2]

table = NamedArray(zeros(16, 7))

typeof(names(X)[index][1])

for i in 1:16
    for j in 1:6
        table[i,j+1] = GLM.coeftable(Lasso_HDM).cols[j][i]
    end
end

T = DataFrame(table, [ :"Variable", :"Estimate", :"Std.error", :"t", :"P-value", :"Lower-bound", :"Upper-bound"])

T[!,:Variable] = string.(T[!,:Variable])  # string - first column 

for i in 1:16

T[i,1] = names(X)[index][i]

end


header = (["Variable", "Estimate", "Std.error", "t-value", "P-value", "Lower-bound", "Upper-bound"])

pretty_table(T; backend = Val(:html), header = header, formatters=ft_round(5), alignment=:c)

xerror = T[!,2] .- T[!,6]

scatter(names(X)[index], T[!,2] , label = "Estimator", yerrors = xerror, 
        ytick = -2:0.05:2, linestyle = :dash, seriescolor=:blue)
plot!(size=(800,650), title="Heterogeneous wage effects - CI", xrotation=90)
hline!( [0], linestyle = :dash , label = "line")














