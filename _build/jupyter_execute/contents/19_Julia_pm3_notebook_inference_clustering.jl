using Pkg
# Pkg.add("CSV"), using CSV
# Pkg.add("DataFrames"), using DataFrames
# Pkg.add("StatsModels"), using StatsModels
# Pkg.add("GLM"), using GLM
# Pkg.add("Random"), using Random

using Pkg, CSV, DataFrames, StatsModels, GLM, Random

data = CSV.File("../data/gun_clean.csv") |> DataFrame;
println("Number of rows: ",size(data,1))
println("Number of columns: ",size(data,2))

#################################  Find Variable Names from Dataset ########################

function varlist(df = nothing , type_dataframe = ["numeric","categorical","string"], pattern=String , exclude =  nothing)

    varrs = []
    if "numeric" in type_dataframe
        append!(varrs, [i for i in names(data) if eltype(eachcol(data)[i]) <: Number])    
    end
    if "categorical" in type_dataframe
        append!(varrs,[i for i in names(data) if eltype(eachcol(data)[i]) <: CategoricalVector])
    end
    if "string" in type_dataframe
        append!(varrs,[i for i in names(data) if eltype(eachcol(data)[i]) <: String])
    end
    varrs[(!varrs in exclude) & varrs[findall(x->contains(x,pattern),names(data))]]
end

################################# Create Variables ###############################

# Dummy Variables for Year and County Fixed Effects
fixed = filter(x->contains(x, "X_Jfips"), names(data));
year = filter(x->contains(x, "X_Tyear"), names(data));

census = []
census_var = ["AGE", "BN", "BP", "BZ", "ED", "EL", "HI", "HS", "INC", "LF", "LN", "PI", "PO", "PP", "PV", "SPR", "VS"]

for i in 1:size(census_var,1) 
    append!(census, filter(x->contains(x, census_var[i]), names(data)))
end

################################ Variables ##################################

# Treatment Variable
d = ["logfssl"];

# Outcome Variable
y = ["logghomr"];

# Other Control Variables
X1 = ["logrobr", "logburg", "burg_missing", "robrate_missing"];
X2 = ["newblack", "newfhh", "newmove", "newdens", "newmal"];

#################################  Partial out Fixed Effects ########################

# Variables to be Partialled-out
variable = [y, d,X1, X2, census]
varlis = []

# Partial out Variables in varlist from year and county fixed effect
for i in variable
    append!(varlis,i)
end

# Running the following lines takes aprox. 10 minutes (depends on your CPU)

example = DataFrame(CountyCode = data[:,"CountyCode"]);
rdata = DataFrame(CountyCode = data[:,"CountyCode"]);

for i in 1:size(varlis,1)
    rdata[!,varlis[i]]= residuals(lm(term(Symbol(varlis[i])) ~ sum(term.(Symbol.(year))) + sum(term.(Symbol.(fixed))), data))
end

first(rdata, 6)

# load dataset
rdata_read = CSV.File("../data/gun_clean2.csv") |> DataFrame
data_1 = rdata_read[!, names(rdata)]
n = size(data_1,1)

column_names = names(data_1)
result = []

for i in 1:size(data_1,1)
    for j in 1:size(data_1,2)
        data_1[i,j] = round(data_1[i,j], digits=6)
        rdata[i,j] = round(rdata[i,j], digits=6)
    end
end

for col in column_names
    result = sum(data_1[!,col] .== rdata[!,col])
end

# Treatment variable
D = rdata[!,d]

# Outcome variable
Y = rdata[!,y];

# Construct matrix Z
Z = rdata[!, varlis[3:end]];


clu = select(rdata,:CountyCode)
data = hcat(Y,D,Z,clu);
first(data, 6)

size(data), size(rdata)

using FixedEffectModels

# OLS clustering at the County level

fm_1 = @formula(logghomr ~ 0 + logfssl + fe(CountyCode))
baseline_ols = reg(data, fm_1, Vcov.cluster(:CountyCode))

println("2.5% : ", GLM.coeftable(baseline_ols).cols[5])
println("97.5% : " , GLM.coeftable(baseline_ols).cols[6])
println("Estimate: ", GLM.coeftable(baseline_ols).cols[1])
println("Cluster s.e. : " , GLM.r2(baseline_ols))
println("T-value : ", GLM.coeftable(baseline_ols).cols[3])
println("Pr(>|t|) : " , GLM.coeftable(baseline_ols).cols[4])

control_formula = term(:logghomr) ~ term(:logfssl) + sum(term.(Symbol.(names(Z)))) + fe(:CountyCode)
control_ols = reg(data, control_formula)

println("For <<logfssl>> variable: ")
println("2.5% : ", GLM.coeftable(control_ols).cols[5][1])
println("97.5% : " , GLM.coeftable(control_ols).cols[6][1])
println("Estimate: ", GLM.coeftable(control_ols).cols[1][1])
println("Cluster s.e. : " , GLM.r2(control_ols))
println("T-value : ", GLM.coeftable(control_ols).cols[3][1])
println("Pr(>|t|) : " , GLM.coeftable(control_ols).cols[4][1])

using  MLDataUtils, MLBase

function DML2_for_PLM(z , d , y, dreg , yreg , nfold, clu)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices 
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        
        # Lasso regression, excluding folds selected 
        dfit = dreg(z[foldid[i],:], d[foldid[i]])
        yfit = yreg(z[foldid[i],:], y[foldid[i]])
        
        # Predict estimates using the 
        dhat = GLM.predict(dfit, z[Not(foldid[i]),:])
        yhat = GLM.predict(yfit, z[Not(foldid[i]),:])
        
        # Save errors 
        dtil[Not(foldid[i])] = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])] = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil, clu=clu)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ dtil +fe(clu)))
    coef_est = coef(rfit)[1]
    se = FixedEffectModels.coeftable(rfit).cols[2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

# Create main variables
z = Matrix(Z);
d = D[!,1];
y = Y[!,1];
clu = rdata[!, :CountyCode];
first(DataFrame(logghomr = y,logfssl = d,CountyCode = clu ),6)

using Lasso

Random.seed!(123)
dreg(z,d) = fit(LassoModel, z, d, standardize = false)
yreg(z,y) = fit(LassoModel, z, y, standardize = false)
DML2_lasso = DML2_for_PLM(z, d, y, dreg, yreg, 10, clu);

include("hdmjl/hdmjl.jl")

function DML2_lasso_cv_hdm(z , d , y, dreg , yreg , nfold, clu)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        
        # Lasso regression, excluding folds selected 
        dtil= dreg(Z[foldid[i],:], D[foldid[i],:])
        ytil= dreg(Z[foldid[i],:], Y[foldid[i],:])
        
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ 0 + dtil))
    coef_est = coef(rfit)[1]
    se = FixedEffectModels.coeftable(rfit).cols[2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

function dreg(Z, Y)
    res_Y_0 = rlasso_arg( Z, Y, nothing, true, true, true, false, false, 
                    nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )
    res_Y = rlasso(res_Y_0)["residuals"]
end

function yreg(Z, D)
    res_D_0 = rlasso_arg( Z, D, nothing, true, true, true, false, false, 
                        nothing, 1.1, nothing, 5000, 15, 10^(-5), -Inf, true, Inf, true )
    res_D = rlasso(res_D_0)["residuals"]
end

DML2_lasso_hdm = DML2_lasso_cv_hdm(z, d, y, dreg, yreg, 10, clu);

using GLMNet

function DML2_lasso_cv(z , d , y, dreg , yreg , nfold, clu)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(z[foldid[i],:], d[foldid[i]])
        yfit = yreg(z[foldid[i],:], y[foldid[i]])
        dhat = GLMNet.predict(dfit, z[Not(foldid[i]),:])
        yhat = GLMNet.predict(yfit, z[Not(foldid[i]),:])
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil, clu=clu)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ dtil +fe(clu)))
    coef_est = coef(rfit)[1]
    se = FixedEffectModels.coeftable(rfit).cols[2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

##ML method = lasso from glmnet 
dreg(z, d) = glmnetcv(z, d, alpha = 1)    
yreg(z, y) = glmnetcv(z, y, alpha = 1)   

DML2_lasso_cv_1 = DML2_lasso_cv(z, d, y, dreg, yreg, 10, clu);

##ML method = elastic net from glmnet 
dreg(z, d) = glmnetcv(z, d, alpha = 0.5) 
yreg(z, y) = glmnetcv(z, y, alpha = 0.5)

DML2_elnet =  DML2_lasso_cv(z, d, y, dreg, yreg, 10, clu);

##ML method = elastic net from glmnet 
dreg(z, d) = glmnetcv(z, d, alpha = 0) 
yreg(z, y) = glmnetcv(z, y, alpha = 0)

DML2_ridge = DML2_lasso_cv(z, d, y, dreg, yreg, 10, clu);

Random.seed!(123)
dreg(z,d) = lm(z,d)
yreg(z,y) = lm(z,y)
DML2_ols = DML2_for_PLM(z, d, y, dreg, yreg, 10, clu);

import Pkg; Pkg.add("MLJ")
import Pkg; Pkg.add("DecisionTree")

using DecisionTree, MLJ

function DML2_RF(z , d , y, dreg , yreg , nfold, clu)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    println("Folds: " )
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(z[foldid[i],:], d[foldid[i]])
        yfit = yreg(z[foldid[i],:], y[foldid[i]])
        dhat = apply_forest(dfit,z[Not(foldid[1]),:])
        yhat = apply_forest(yfit,z[Not(foldid[1]),:])
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
        println(i)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil, clu=clu)
    
    # OLS clustering at the County level
    rfit = reg(data, @formula(ytil ~ dtil +fe(clu)))
    coef_est = coef(rfit)[1]
    se = FixedEffectModels.coeftable(rfit).cols[2]

    println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

function dreg(z,d)
    RFmodel = build_forest(d,z)
end
function yreg(z,y)
    RFmodel = build_forest(y,z)
end

DML2_RF_1 = DML2_RF(z, d, y, dreg, yreg, 10, clu);

using PrettyTables

mods = [DML2_ols, DML2_lasso, DML2_lasso_cv_1, DML2_ridge, DML2_elnet, DML2_RF_1];
mods_name = ["DML2_ols", "DML2_lasso", "DML2_lasso_cv", "DML2_ridge", "DML2_elnet", "DML2_RF"];

RMSE_Y = []
RMSE_D = []

for i in mods
    push!(RMSE_Y, sqrt(mean(i[2][!,1])^2))
    push!(RMSE_D, sqrt(mean(i[2][!,2])^2))
end

result = DataFrame([mods_name RMSE_Y RMSE_D], [:Models, :RMSE_Y, :RMSE_D])
pretty_table(result; formatters = ft_printf("%5.10f"))

#DML with cross-validated Lasso:
dreg(z,d) = glmnetcv(z,d, alpha = 1)
yreg(z,y) = glmnetcv(z,y, alpha = 0)

DML2_best = DML2_lasso_cv(z, d, y, dreg, yreg, 10, clu);

ols_coef = GLM.coeftable(baseline_ols).cols[1][1]
ols_std = GLM.coeftable(baseline_ols).cols[2][1]
control_ols_coef = GLM.coeftable(control_ols).cols[1][1]
control_ols_std = GLM.coeftable(control_ols).cols[2][1]
lasso_coef = GLM.coeftable(DML2_lasso[1]).cols[1][1]
lasso_std = GLM.coeftable(DML2_lasso[1]).cols[2][1]
DML2_lasso_cv_1_coef = GLM.coeftable(DML2_lasso_cv_1[1]).cols[1][1]
DML2_lasso_cv_1_std = GLM.coeftable(DML2_lasso_cv_1[1]).cols[2][1]
DML2_elnet_coef = GLM.coeftable(DML2_elnet[1]).cols[1][1]
DML2_elnet_std = GLM.coeftable(DML2_elnet[1]).cols[2][1]
DML2_ridge_coef = GLM.coeftable(DML2_ridge[1]).cols[1][1]
DML2_ridge_std = GLM.coeftable(DML2_ridge[1]).cols[2][1]
DML2_RF_1_coef = GLM.coeftable(DML2_RF_1[1]).cols[1][1]
DML2_RF_1_std = GLM.coeftable(DML2_RF_1[1]).cols[2][1]
DML2_best_coef = GLM.coeftable(DML2_best).cols[1][1]
DML2_best_std = GLM.coeftable(DML2_best).cols[2][1];

tabla = DataFrame(modelos = ["Baseline OLS", "Least Squares with controls", "Lasso", "CV Lasso", "CV Elnet", "CV Ridge", "Random Forest", "Best"], 
Estimate = [ols_coef, control_ols_coef, lasso_coef, DML2_lasso_cv_1_coef, DML2_elnet_coef, DML2_ridge_coef, DML2_RF_1_coef, DML2_best_coef], 
StdError = [ols_std, control_ols_std, lasso_std, DML2_lasso_cv_1_std, DML2_elnet_std, DML2_ridge_std, DML2_RF_1_std, DML2_best_std])
