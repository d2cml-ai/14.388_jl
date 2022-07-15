# Import relevant packages
using Pkg
Pkg.add("CSV"), using CSV
Pkg.add("DataFrames"), using DataFrames
Pkg.add("GLM"), using GLM
Pkg.add("FixedEffectModels"), using FixedEffectModels
Pkg.add("DecisionTree"), using DecisionTree
Pkg.add("PrettyTables"), using PrettyTables
Pkg.add("CovarianceMatrices"), using CovarianceMatrices
Pkg.add("RegressionTables"), using RegressionTables
Pkg.add("StatsFuns"), using StatsFuns
Pkg.add("Plots"), using Plots
Pkg.add("RData"), using RData
Pkg.add("MLBase"), using MLBase

# load data
rdata_read = RData.load("../data/ajr.RData")
AJR = rdata_read["AJR"]
names(AJR)
println("Number of Rows : ", size(AJR)[1],"\n","Number of Columns : ", size(AJR)[2],) #rows and columns

first(AJR, 5)

function DML2_for_PLIVM(x , d , z, y, dreg , yreg , zreg, nfold)
     # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices 
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    ztil = ones(nobser);

    println("Folds: " )

    for i in 1:nfold
        
        # Lasso regression, excluding folds selected 
        dfit = dreg(x[foldid[i],:], d[foldid[i]])
        zfit = zreg(x[foldid[i],:], z[foldid[i]])
        yfit = yreg(x[foldid[i],:], y[foldid[i]])
        
        # Predict estimates using the 
        dhat = apply_forest(dfit,x[Not(foldid[i]),:])
        zhat = apply_forest(zfit,x[Not(foldid[i]),:])
        yhat = apply_forest(yfit,x[Not(foldid[i]),:])

        # Save errors 
        dtil[Not(foldid[i])] = (d[Not(foldid[i])] - dhat)
        ztil[Not(foldid[i])] = (z[Not(foldid[i])] - zhat)
        ytil[Not(foldid[i])] = (y[Not(foldid[i])] - yhat)
        println(i)

    end

    data = DataFrame(ytil = ytil, dtil = dtil, ztil = ztil)
    ivfit = reg(data, @formula(ytil ~  + (dtil ~ ztil)))
    # OLS clustering at the County level
    coef_est =  ivfit.coef[2]
    se = FixedEffectModels.coeftable(ivfit).cols[2][1]
    println( "\n Coefficient (se) =", coef_est, "(",se,")" )
    
    Final_result = ("coef_est" => coef_est , "se" => se , "dtil" => dtil , "ytil" => ytil , "ztil" => ztil);
       
end

y = AJR[!,"GDP"]
d = AJR[!,"Exprop"]
z = AJR[!,"logMort"];

Y = DataFrame(GDP = y)
D = DataFrame(Exprop = d)
Z = DataFrame(logMort = z);

xraw_formula = @formula(GDP ~ Latitude+ Africa+Asia + Namer + Samer)
xraw_dframe = ModelFrame(xraw_formula, AJR)
xraw1  = ModelMatrix(xraw_dframe)
xraw = xraw1.m
size(xraw)

x_formula = @formula(GDP ~ -1 + Latitude + Latitude2 + Africa + Asia + Namer + Samer
    + Latitude*Latitude2 + Latitude*Africa + Latitude*Asia + Latitude*Namer + Latitude*Samer
    + Latitude2*Africa + Latitude2*Asia + Latitude2*Namer + Latitude2*Samer
    + Africa*Asia + Africa*Namer + Africa*Samer
    + Asia*Namer + Asia*Samer
    + Namer*Samer)
x_dframe = ModelFrame( x_formula, AJR)
x1 = ModelMatrix(x_dframe)
x = x1.m
size(x1)

X = DataFrame(Latitude = x_dframe.data[2], Latitude2 = x_dframe.data[3], Africa = x_dframe.data[4],
    Asia = Africa = x_dframe.data[5], Namer = Africa = x_dframe.data[6], Samer = Africa = x_dframe.data[7])
size(X)

y_model = DataFrame(y_model = y);

function dreg( x_1 , d_1 )
    
    if d_1 != nothing && ( typeof(d_1) != String )
        mtry1 = convert(Int64,findmax(round( ( size(x_1)[ 2 ]/3 ), digits = 0))[1])
    else
        mtry1 = convert(Int64,round(sqrt( size(x)[ 2 ] ), digits = 0))
    end
    
    if d_1 != nothing && ( typeof(d_1) != String )
        nodesize1 = 5
    else
        nodesize1 = 1
    end
    n_subfeatures=-1; n_trees=10; partial_sampling=0.7; max_depth=-1
    min_samples_leaf= nodesize1; min_samples_split=2; min_purity_increase=0.0; seed=0
    
    RFmodel =  build_forest(d_1, x_1, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase; rng = seed)
end

function yreg( x_1, y_1 )
    
    if y_1 != nothing && ( typeof(y_1) != String )
        mtry1 = convert(Int64,findmax(round( ( size(x_1)[ 2 ]/3 ), digits = 0))[1])
    else
        mtry1 = convert(Int64,round(sqrt( size(x_1)[ 2 ] ), digits = 0))
    end
    
    if y_1 != nothing && ( typeof(y_1) != String )
        nodesize1 = 5
    else
        nodesize1 = 1
    end
    n_subfeatures=-1; n_trees=10; partial_sampling=0.7; max_depth=-1
    min_samples_leaf= nodesize1; min_samples_split=2; min_purity_increase=0.0; seed=0
    
    RFmodel =  build_forest(y_1, x_1, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase; rng = seed)
end
                
function zreg( x_1, z_1 )
    
    if z_1 != nothing && ( typeof(z_1) != String )
        mtry1 = convert(Int64,findmax(round( ( size(x_1)[ 2 ]/3 ), digits = 0))[1])
    else
        mtry1 = convert(Int64,round(sqrt( size(x_1)[ 2 ] ), digits = 0))
    end
    
    if z_1 != nothing && ( typeof(z_1) != String )
        nodesize1 = 5
    else
        nodesize1 = 1
    end
    n_subfeatures=-1; n_trees=10; partial_sampling=0.7; max_depth=-1
    min_samples_leaf= nodesize1; min_samples_split=2; min_purity_increase=0.0; seed=0
    
    RFmodel =  build_forest(z_1, x_1, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase; rng = seed)
end

DML2_RF = DML2_for_PLIVM(xraw, d, z, y, dreg, yreg, zreg, 20)

print( "\n DML with Post-Lasso \n" )

include("E:/causal_ml/hdmjl/hdmjl.jl")

#include("hdmjl/hdmjl.jl")

function DML2_lasso_cv(x , d , z, y, dreg , yreg , zreg, nfold)
     # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices 
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    ztil = ones(nobser);

    println("Folds: " )

    for i in 1:nfold

        # Save errors 
        # Lasso regression, excluding folds selected 
        dtil= dreg(X[foldid[i],:], D[foldid[i],:])
        ztil= dreg(X[foldid[i],:], Z[foldid[i],:])
        ytil= dreg(X[foldid[i],:], Y[foldid[i],:])
        println(i)

    end

    data = DataFrame(ytil = ytil, dtil = dtil, ztil = ztil)
    
    ivfit = reg(data, @formula(ytil ~  + (dtil ~ ztil)))
    # OLS clustering at the County level
    coef_est =  ivfit.coef[2]
    se = FixedEffectModels.coeftable(ivfit).cols[2][1]
    println( "\n Coefficient (se) =", coef_est, "(",se,")" )
    
    Final_result = ("coef_est" => coef_est , "se" => se , "dtil" => dtil , "ytil" => ytil , "ztil" => ztil);
       
end

function dreg(x_1, d_1)
    res_Y_0 = rlasso_arg( x_1, d_1, nothing, true, true, true, false, false, 
                    nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )
    res_Y = rlasso(res_Y_0)["residuals"]
end

function yreg(x_1, y_1)
    res_D_0 = rlasso_arg( x_1, y_1, nothing, true, true, true, false, false, 
                        nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )
    res_D = rlasso(res_D_0)["residuals"]
end

function yreg(x_1, z_1)
    res_D_0 = rlasso_arg( x_1, z_1, nothing, true, true, true, false, false, 
                        nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )
    res_D = rlasso(res_D_0)["residuals"]
end

DML2_lasso = DML2_lasso_cv(x, d, z, y, dreg, yreg, zreg, 20)

# Compare Forest vs Lasso

mods = [DML2_RF, DML2_lasso];
mods_name = ["DML2_RF", "DML2_lasso"];

RMSE_Y = []
RMSE_D = []
RMSE_Z = []

for i in mods
    push!(RMSE_Y, sqrt(sum(i[4][2].^2)/length(i[4][2])))
    push!(RMSE_D, sqrt(sum(i[3][2].^2)/length(i[3][2])))
    push!(RMSE_Z, sqrt(sum(i[5][2].^2)/length(i[5][2])))
end

result = DataFrame([mods_name RMSE_Y RMSE_D RMSE_Z], [:Models, :RMSE_Y, :RMSE_D, :RMSE_Z])
pretty_table(result; formatters = ft_printf("%5.10f"))

data1 = DataFrame(ytil = DML2_lasso[4][2], dtil = DML2_lasso[3][2], ztil = DML2_lasso[5][2]);

ols1 = fit(LinearModel, @formula(dtil ~ 0 + ztil), data1)

stderror(HC1(), ols1)

data2 = DataFrame(ytil = DML2_RF[4][2], dtil = DML2_RF[3][2], ztil = DML2_RF[5][2]);

ols2 = fit(LinearModel, @formula(dtil ~ 0 + ztil), data2)

stderror(HC1(), ols2)

function DML_AR_PLIV(rY, rD, rZ, grid, alpha)
    n = size(rY)[1]
    Cstat = zeros(size(grid))
    
    for i in 1:length(grid)
        Cstat[i] = n * ((mean((rY - grid[i] * rD).* rZ))^2) / var((rY - grid[i] * rD).* rZ)
    end
    
    data_ar = DataFrame(grid = grid, Cstat = Cstat)
    
    LB = minimum(data_ar[data_ar[:,2] .< chisqinvcdf(1, 1 - alpha), :][!,1])
    UB = maximum(data_ar[data_ar[:,2] .< chisqinvcdf(1, 1 - alpha), :][!,1])
    
    println( "UB =" , UB, "LB ="  ,LB)

    Plots.plot(grid, Cstat, color=:black)
    xlabel!("Effect of institutions")
    ylabel!("Statistic")
    vline!([LB], color=:red, label="1", linestyle=:dash)
    vline!([UB], color=:red, label="2", linestyle=:dash)
    hline!([chisqinvcdf(1, 1 - alpha)], color=:skyblue, linestyle=:dash)
end

DML_AR_PLIV(data1[!,1], data1[!,2], data1[!,3], collect(range( -2, 2.001, step = 0.01 )), 0.05 )

DML_AR_PLIV(data2[!,1], data2[!,2], data2[!,3], collect(range( -2, 2.001, step = 0.01 )), 0.05 )
