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

# using Pkg
# Pkg.add("CSV"), using CSV
# Pkg.add("DataFrames"), using DataFrames
# Pkg.add("StatsModels"), using StatsModels
# Pkg.add("GLM"), using GLM
# Pkg.add("Random"), using Random
# Pkg.add("MLDataUtils"), using MLDataUtils
# Pkg.add("MLBase"), using MLBase
# Pkg.add("FixedEffectModels"), using FixedEffectModels
# Pkg.add("Lasso"), using Lasso
# Pkg.add("MLJ"), using MLJ
# Pkg.add("DecisionTree"), using DecisionTree
# Pkg.add("RData"), using RData
# Pkg.add("GLMNet"), using GLMNet
# Pkg.add("PrettyTables"), using PrettyTables

using Pkg, CSV, DataFrames, StatsModels, GLM, Random, RData, MLDataUtils, MLBase, FixedEffectModels, Lasso, MLJ, DecisionTree, GLMNet, PrettyTables

url = "https://github.com/d2cml-ai/14.388_jl/raw/github_data/data/wage2015_subsample_inference.RData"
download(url, "data.RData")
rdata_read = RData.load("data.RData")
rm("data.RData")
data = rdata_read["data"]
names(data)
println("Number of Rows : ", size(data)[1],"\n","Number of Columns : ", size(data)[2],) #rows and columns

y = GrowthData[!,1]
y= reshape(y, (length(y),1))
d = GrowthData[!,3]
d= reshape(d, (length(y),1))
x = GrowthData[!,4:end]
x = Matrix(x);

println("\n length of y is \n", size(y,1) )
println("\n num features x is \n", size(x,1 ) )

# Naive OLS
print( "\n Naive OLS that uses all features w/o cross-fitting \n" )
fm = term(:Outcome) ~ term(:gdpsh465) +sum(term.(Symbol.(names(GrowthData[:,4:size(GrowthData,2)]))));
lres = reg(GrowthData, fm);
first(DataFrame(GLM.coeftable(lres)))

function DML2_for_PLM(x , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices 
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        
        # Lasso regression, excluding folds selected 
        dfit = dreg(x[foldid[i],:], d[foldid[i]])
        yfit = yreg(x[foldid[i],:], y[foldid[i]]) 
        
        # Predict estimates using the 
        dhat = GLM.predict(dfit, x[Not(foldid[i]),:])
        yhat = GLM.predict(yfit, x[Not(foldid[i]),:])
        
        # Save errors 
        dtil[Not(foldid[i])] = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])] = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ dtil ))
    coef_est = GLM.coef(rfit)[2]
    se = GLM.coeftable(rfit).cols[2][2]
    
    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

#DML with OLS
print( "\n DML with OLS w/o feature selection \n" )

dreg(x, d) = lm(x,vec(d))    
yreg(x, y) = lm(x,vec(y))

DML2_ols = DML2_for_PLM(x, d, y, dreg, yreg, 10 );

function DML2_lasso_cv(x , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(x[foldid[i],:], d[foldid[i]])
        yfit = yreg(x[foldid[i],:], y[foldid[i]])
        
        dhat = GLMNet.predict(dfit, x[Not(foldid[i]),:])
        yhat = GLMNet.predict(yfit, x[Not(foldid[i]),:])
        
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = lm(@formula(ytil ~ dtil), data)
    coef_est = GLM.coef(rfit)[2]
    se = GLM.coeftable(rfit).cols[2][2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

# DML with LASSO
print( "\n DML with Lasso \n" )

##ML method = lasso from glmnet 
dreg(x, d) = glmnetcv(x, d, alpha = 1)    
yreg(x, y) = glmnetcv(x, y, alpha = 1)  
DML2_lasso_cv_1 = DML2_lasso_cv(x, d, y, dreg, yreg, 10);

function DML2_RF(x , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(x[foldid[i],:], d[foldid[i]])
        yfit = yreg(x[foldid[i],:], y[foldid[i]])
        
        dhat = apply_forest(dfit,x[Not(foldid[1]),:])
        yhat = apply_forest(yfit,x[Not(foldid[1]),:])
        
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ dtil)) #unico cambio
    coef_est = GLM.coef(rfit)[2]
    se = GLM.coeftable(rfit).cols[2][2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

print( "\n DML with Random Forest \n" )
function dreg(x,d)
    min_samples_leaf = 5
    rng = 3
    RFmodel = build_forest(d,x, min_samples_leaf, rng)
end
function yreg(x,y)
    min_samples_leaf = 5
    rng = 3
    RFmodel = build_forest(y,x, min_samples_leaf, rng)
end

DML2_RF_1 = DML2_RF(x, d, y, dreg, yreg, 5);

function DML2_lasso_RF(x , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(x[foldid[i],:], d[foldid[i]])
        yfit = yreg(x[foldid[i],:], y[foldid[i]])
        
        dhat = GLMNet.predict(dfit,x[Not(foldid[1]),:])
        yhat = apply_forest(yfit,x[Not(foldid[1]),:])
        
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ dtil)) #unico cambio
    coef_est = GLM.coef(rfit)[2]
    se = GLM.coeftable(rfit).cols[2][2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

print( "\n DML with Lasso/Random Forest \n" )

dreg(x, d) = glmnetcv(x, d, alpha = 1)

    min_samples_leaf = 5
    rng = 3
yreg(x,y) = build_forest(y,x, min_samples_leaf, rng)

DML2_lasso_RF_1 = DML2_lasso_RF(x , d , y, dreg , yreg , 2);

mods = [DML2_ols, DML2_lasso_cv_1, DML2_RF_1];
mods_name = ["DML2_ols", "DML2_lasso", "DML2_RF"];

RMSE_Y = []
RMSE_D = []

for i in mods
    push!(RMSE_Y, sqrt(sum(i[2][!,1].^2)/length(i[2][!,1])))
    push!(RMSE_D,sqrt(sum(i[2][!,2].^2)/length(i[2][!,2])))
end

result = DataFrame([mods_name RMSE_Y RMSE_D], [:Models, :RMSE_Y, :RMSE_D])
pretty_table(result; formatters = ft_printf("%5.10f"))
