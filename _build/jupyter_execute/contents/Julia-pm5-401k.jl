using RData, CSV, DataFrames, StatsBase, Gadfly, StatsModels, GLM

rdata = RData.load("../data/pension.RData", convert = true);

data = DataFrame(rdata["pension"]);
categorical!(data, [:e401, :p401]);

data[!, "n"] .= 1;
e_count = combine(groupby(data, :e401), :n => sum => :Freq);
plot(e_count, color=:e401, y=:Freq, Geom.bar(position=:dodge))

plot(data, x=:net_tfa, xgroup=:e401, color=:e401, Geom.subplot_grid(Geom.density()))

e1 = data[data.e401 .== 0, :];
e2 = data[data.e401 .== 1, :];
round(mean(e2.net_tfa) - mean(e1.net_tfa))

p1 = data[data.p401 .== 0, :];
p2 = data[data.p401 .== 1, :];
round(mean(p2.net_tfa) - mean(p1.net_tfa))

# We define the `poly` function as provided by the documentation of the `StatsModels` package:

# syntax: best practice to define a _new_ function
poly(x, n) = x^n

# type of model where syntax applies: here this applies to any model type
const POLY_CONTEXT = Any

# struct for behavior
struct PolyTerm{T,D} <: AbstractTerm
    term::T
    deg::D
end

Base.show(io::IO, p::PolyTerm) = print(io, "poly($(p.term), $(p.deg))")

# for `poly` use at run-time (outside @formula), return a schema-less PolyTerm
poly(t::Symbol, d::Int) = PolyTerm(term(t), term(d))

# for `poly` use inside @formula: create a schemaless PolyTerm and apply_schema
function StatsModels.apply_schema(t::FunctionTerm{typeof(poly)},
                                  sch::StatsModels.Schema,
                                  Mod::Type{<:POLY_CONTEXT})
    apply_schema(PolyTerm(t.args_parsed...), sch, Mod)
end

# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::PolyTerm,
                                  sch::StatsModels.Schema,
                                  Mod::Type{<:POLY_CONTEXT})
    term = apply_schema(t.term, sch, Mod)
    isa(term, ContinuousTerm) ||
        throw(ArgumentError("PolyTerm only works with continuous terms (got $term)"))
    isa(t.deg, ConstantTerm) ||
        throw(ArgumentError("PolyTerm degree must be a number (got $t.deg)"))
    PolyTerm(term, t.deg.n)
end

function StatsModels.modelcols(p::PolyTerm, d::NamedTuple)
    col = modelcols(p.term, d)
    reduce(hcat, [col.^n for n in 1:p.deg])
end

# the basic terms contained within a PolyTerm (for schema extraction)
StatsModels.terms(p::PolyTerm) = terms(p.term)
# names variables from the data that a PolyTerm relies on
StatsModels.termvars(p::PolyTerm) = StatsModels.termvars(p.term)
# number of columns in the matrix this term produces
StatsModels.width(p::PolyTerm) = p.deg

StatsBase.coefnames(p::PolyTerm) = coefnames(p.term) .* "^" .* string.(1:p.deg)

# output


#Constructing the Data

formula_flex = @formula(net_tfa ~ e401 + poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize, 2) + marr + twoearn + db + pira + hown);
formula_flex = apply_schema(formula_flex, schema(data));
y, x = modelcols(formula_flex, data);
y = Float64.(y)
d = x[:, 1];
x = x[:, Not(1)];
size(x, 2)

using MLDataUtils, MLBase, Random, FixedEffectModels, GLMNet

function DML2_for_PLM(x , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(x,1)
    
    # Define folds indices 
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    
    dl = convert(Matrix{Float64}, [(d .< 0.5) (d .>= 0.5)])
    
    # loop to save results
    for i in 1:nfold
        
        # Lasso regression, excluding folds selected 
        dfit = dreg(x[foldid[i],:], dl[foldid[i], :])
        yfit = yreg(x[foldid[i],:], y[foldid[i]])
        
        # Predict estimates using the 
        dhat = GLMNet.predict(dfit, x[Not(foldid[i]),:], outtype = :prob)
        yhat = GLMNet.predict(yfit, x[Not(foldid[i]),:])
        
        # Save errors 
        dtil[Not(foldid[i])] = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])] = (y[Not(foldid[i])] - yhat)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = fit(LinearModel, reshape(dtil, nobser, 1), ytil)
    # coef_est = coef(rfit)[2]
    # se = FixedEffectModels.coeftable(rfit).cols[2][2]

    # println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

# Estimating the PLR

Random.seed!(123)
dreg(x,d) = glmnetcv(x, d, nfolds = 5, Binomial())
yreg(x,y) = glmnetcv(x, y, nfolds = 5)
lasso_fit, lasso_data = DML2_for_PLM(x, d, y, dreg, yreg, 3);
lasso_fit

# cross-fitted RMSE: outcome
lasso_y_rmse = sqrt(mean((lasso_data[!, 1] .- StatsBase.coef(lasso_fit)[1] * lasso_data[!, 2]) .^ 2))

# cross-fitted RMSE: treatment

lasso_d_rmse = sqrt(mean(lasso_data[!, 2] .^ 2));

println(lasso_d_rmse)

# cross-fitted ce: treatment

mean(ifelse.(d .- lasso_data[!, 2] .> 0.5, 1, 0) .!= d)

# Random Forrest

using DecisionTree

function DML2_RF(z , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(z[foldid[i],:], d[foldid[i]])
        yfit = yreg(z[foldid[i],:], y[foldid[i]])
        dhat = apply_forest(dfit,z[Not(foldid[i]),:])
        yhat = apply_forest(yfit,z[Not(foldid[i]),:])
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    # rfit = reg(data, @formula(ytil ~ dtil))
    rfit = fit(LinearModel, reshape(dtil, nobser, 1), ytil)
    # coef_est = coef(rfit)[1]
    # se = FixedEffectModels.coeftable(rfit).cols[2]

    # println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

foldid = collect(Kfold(size(x)[1], 3))

size(foldid[1], 1)

Random.seed!(123)

dreg(x, d) = build_forest(d, x)
yreg(x, y) = build_forest(y, x)

rf_fit, rf_data = DML2_RF(x, d, y, dreg, yreg, 3);
rf_fit

rf_y_rmse = sqrt(mean((rf_data[!, 1] .- StatsBase.coef(rf_fit)[1] * rf_data[!, 2]) .^ 2))
rf_d_rmse = sqrt(mean(rf_data[!, 2] .^ 2))

println(rf_y_rmse)

println(rf_d_rmse)

mean(ifelse.(d .- rf_data[!, 2] .> 0.5, 1, 0) .!= d)

# Trees

function DML2_Tree(z , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(z[foldid[i],:], d[foldid[i]])
        yfit = yreg(z[foldid[i],:], y[foldid[i]])
        dhat = apply_tree(dfit,z[Not(foldid[i]),:])
        yhat = apply_tree(yfit,z[Not(foldid[i]),:])
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = fit(LinearModel, reshape(dtil, nobser, 1), ytil)
    # coef_est = coef(rfit)[1]
    # se = FixedEffectModels.coeftable(rfit).cols[2]

    # println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

Random.seed!(123)

dreg(x, d) = build_tree(d, x, 0, 30, 7, 20, 0.01)
yreg(x, y) = build_tree(y, x, 0, 30, 7, 20, 0.01)

tree_fit, tree_data = DML2_Tree(x, d, y, dreg, yreg, 3);
tree_fit

tree_y_rmse = sqrt(mean((tree_data[!, 1] .- StatsBase.coef(tree_fit)[1] * tree_data[!, 2]) .^ 2))
tree_d_rmse = sqrt(mean(tree_data[!, 2] .^ 2))

println(tree_y_rmse)

println(tree_d_rmse)

mean(ifelse.(d .- tree_data[!, 2] .> 0.5, 1, 0) .!= d)

# Boosting

using XGBoost

function DML2_Boost(z , d , y, dreg , yreg , nfold)
    
    # Num ob observations
    nobser = size(z,1)
    
    # Define folds indices
    foldid = collect(Kfold(size(z)[1], nfold))
    
    # Create array to save errors 
    ytil = ones(nobser)
    dtil = ones(nobser)
    
    # loop to save results
    for i in 1:nfold
        dfit = dreg(z[foldid[i], :], d[foldid[i]])
        yfit = yreg(z[foldid[i], :], y[foldid[i]])
        dhat = XGBoost.predict(dfit, z[Not(foldid[i]), :])
        yhat = XGBoost.predict(yfit, z[Not(foldid[i]), :])
        dtil[Not(foldid[i])]   = (d[Not(foldid[i])] - dhat)
        ytil[Not(foldid[i])]   = (y[Not(foldid[i])] - yhat)
    end
    
    # Create dataframe 
    data = DataFrame(ytil = ytil, dtil = dtil)
    
    # OLS clustering at the County level
    rfit = fit(LinearModel, reshape(dtil, nobser, 1), ytil)
    # coef_est = coef(rfit)[1]
    # se = FixedEffectModels.coeftable(rfit).cols[2]

    # println(" coef (se) = ", coef_est ,"(",se,")")
    
    return rfit, data;
    
end

Random.seed!(123)

dreg(x, d) = xgboost(x, 5, label = d, objective = "binary:logistic", eval_metric = "logloss");
yreg(x, y) = xgboost(x, 5, label = y);

boost_fit, boost_data = DML2_Boost(x, d, y, dreg, yreg, 3);
boost_fit

boost_y_rmse = sqrt(mean((boost_data[!, 1] .- StatsBase.coef(boost_fit)[1] * boost_data[!, 2]) .^ 2))
boost_d_rmse = sqrt(mean(boost_data[!, 2] .^ 2))

println(boost_y_rmse)

println(boost_d_rmse)

mean(ifelse.(d .- boost_data[!, 2] .> 0.5, 1, 0) .!= d)

DataFrame(Statistic = ["Estimate", "Std.Error", "RMSE Y", "RMSE D"], 
    Lasso = [StatsBase.coef(lasso_fit)[1], sqrt(vcov(lasso_fit)[1]), lasso_y_rmse, lasso_d_rmse], 
    RF = [StatsBase.coef(rf_fit)[1], sqrt(vcov(rf_fit)[1]), rf_y_rmse, rf_d_rmse], 
    Trees = [StatsBase.coef(tree_fit)[1], sqrt(vcov(tree_fit)[1]), tree_y_rmse, tree_d_rmse], 
    Boosting = [StatsBase.coef(boost_fit)[1], sqrt(vcov(boost_fit)[1]), boost_y_rmse, boost_d_rmse])

# Function based off of: https://github.com/DoubleML/doubleml-for-r/blob/f00d62c722a2b1e37c01b7f7f772e9d07f452a98/R/double_ml_irm.R
#                        https://github.com/DoubleML/doubleml-for-r/blob/f00d62c722a2b1e37c01b7f7f772e9d07f452a98/R/double_ml_plr.R

# Function takes x, y, d, learners, and nfolds

function IRM_Lasso(x, y, d, ml_g, ml_m, nfold, trimming_threshold = 1e-12)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y1_hat = ones(nobser)
    y0_hat = ones(nobser)
    d_hat = ones(nobser)
    dl = convert(Matrix{Float64}, [(d .< 0.5) (d .>= 0.5)])
    
    # Apply learners to y_0, y_1 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), d[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        m_hat = ml_m(x[foldid[i], :], dl[foldid[i], :])
        
        # Predict: g0_hat, g1_hat, m_hat
        d_hat[Not(foldid[i])] = GLMNet.predict(m_hat, x[Not(foldid[i]), :], outtype = :prob)
        y0_hat[Not(foldid[i])] = GLMNet.predict(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = GLMNet.predict(g1_hat, x[Not(foldid[i]), :])
    
    end
    
    # Residuals: u0_hat, u1_hat, no need for residual in d
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    
    # Trimming
    d_hat[d_hat .< trimming_threshold] .= trimming_threshold
    d_hat[d_hat .> (1 - trimming_threshold)] .= 1 - trimming_threshold

    # Compute regression terms:
    # Left side: y1_hat - y0_hat + d * u1_hat / m_hat - (1 - d) * u0_hat / (1 - m_hat)
    psi_b = y1_hat .- y0_hat .+ d .* u1_hat ./ d_hat - (1 .- d) .* u0_hat ./ (1 .- d_hat)
    
    # Right side: All ones
    psi_a = reshape(ones(nobser), nobser, 1)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, psi_a, psi_b)
    
    # Generate data matrix for output
    u_hat = d .* u1_hat + (1 .- d) .* u0_hat
    d_til = d .- d_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data;
    
end

Random.seed!(123)
ml_m(x, d) = glmnetcv(x, d, nfolds = 5, Binomial())
ml_g(x, y) = glmnetcv(x, y, nfolds = 5)
lasso_fit, lasso_data = IRM_Lasso(x, y, d, ml_g, ml_m, 3, 0.01);
lasso_fit

function IRM_Forest(x, y, d, ml_g, ml_m, nfold)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y1_hat = ones(nobser)
    y0_hat = ones(nobser)
    d_hat = ones(nobser)
    
    # Apply learners to y_0, y_1 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), d[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        m_hat = ml_m(x[foldid[i], :], d[foldid[i]])
        
        # Predict: g0_hat, g1_hat, m_hat
        d_hat[Not(foldid[i])] = apply_forest(m_hat, x[Not(foldid[i]), :])
        y0_hat[Not(foldid[i])] = apply_forest(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = apply_forest(g1_hat, x[Not(foldid[i]), :])
    
    end
    
    # Residuals: u0_hat, u1_hat, no need for residual in d
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    
    # Compute regression terms:
    # Left side: y1_hat - y0_hat + d * u1_hat / m_hat 
    #            - (1 - d) * u0_hat / (1 - m_hat)
    psi_b = y1_hat .- y0_hat .+ d .* u1_hat ./ d_hat - (1 .- d) .* u0_hat ./ (1 .- d_hat)
    
    # Right side: All ones
    psi_a = reshape(ones(nobser), nobser, 1)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, psi_a, psi_b)
    
    # Generate data matrix for output
    u_hat = d .* u1_hat + (1 .- d) .* u0_hat
    d_til = d .- d_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data, psi_a, psi_b;
    
end

Random.seed!(123)

ml_m(x, d) = build_forest(d, x)
ml_g(x, y) = build_forest(y, x)

rf_fit, rf_data, psi_a, psi_b = IRM_Forest(x, y, d, ml_g, ml_m, 3);
rf_fit

psi_b

function IRM_Tree(x, y, d, ml_g, ml_m, nfold)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y1_hat = ones(nobser)
    y0_hat = ones(nobser)
    d_hat = ones(nobser)
    
    # Apply learners to y_0, y_1 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), d[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        m_hat = ml_m(x[foldid[i], :], d[foldid[i]])
        
        # Predict: g0_hat, g1_hat, m_hat
        d_hat[Not(foldid[i])] = apply_tree(m_hat, x[Not(foldid[i]), :])
        y0_hat[Not(foldid[i])] = apply_tree(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = apply_tree(g1_hat, x[Not(foldid[i]), :])
    
    end
    
    # Residuals: u0_hat, u1_hat, no need for residual in d
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    
    # Compute regression terms:
    # Left side: y1_hat - y0_hat + d * u1_hat / m_hat 
    #            - (1 - d) * u0_hat / (1 - m_hat)
    psi_b = y1_hat .- y0_hat .+ d .* u1_hat ./ d_hat - (1 .- d) .* u0_hat ./ (1 .- d_hat)
    
    # Right side: All ones
    psi_a = reshape(ones(nobser), nobser, 1)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, psi_a, psi_b)
    
    # Generate data matrix for output
    u_hat = d .* u1_hat + (1 .- d) .* u0_hat
    d_til = d .- d_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data;
    
end

Random.seed!(123)

ml_m(x, d) = build_tree(d, x, 0, 30, 7, 20, 0.01)
ml_g(x, y) = build_tree(y, x, 0, 30, 7, 20, 0.01)

tree_fit, tree_data = IRM_Tree(x, y, d, ml_g, ml_m, 3);
tree_fit


function IRM_Boost(x, y, d, ml_g, ml_m, nfold, trimming_threshold = 1e-12)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y1_hat = ones(nobser)
    y0_hat = ones(nobser)
    d_hat = ones(nobser)
    
    # Apply learners to y_0, y_1 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), d[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        m_hat = ml_m(x[foldid[i], :], d[foldid[i]])
        
        # Predict: g0_hat, g1_hat, m_hat
        d_hat[Not(foldid[i])] = XGBoost.predict(m_hat, x[Not(foldid[i]), :])
        y0_hat[Not(foldid[i])] = XGBoost.predict(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = XGBoost.predict(g1_hat, x[Not(foldid[i]), :])
    
    end
    
    # Residuals: u0_hat, u1_hat, no need for residual in d
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    
    # Trimming
    d_hat[d_hat .< trimming_threshold] .= trimming_threshold
    d_hat[d_hat .> (1 - trimming_threshold)] .= 1 - trimming_threshold

    # Compute regression terms:
    # Left side: y1_hat - y0_hat + d * u1_hat / m_hat 
    #            - (1 - d) * u0_hat / (1 - m_hat)
    psi_b = y1_hat .- y0_hat .+ d .* u1_hat ./ d_hat - (1 .- d) .* u0_hat ./ (1 .- d_hat)
    
    # Right side: All ones
    psi_a = reshape(ones(nobser), nobser, 1)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, psi_a, psi_b)
    
    # Generate data matrix for output
    u_hat = d .* u1_hat + (1 .- d) .* u0_hat
    d_til = d .- d_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data, psi_a, psi_b;
    
end

Random.seed!(123)

ml_m(x, d) = xgboost(x, 5, label = d, objective = "binary:logistic", eval_metric = "logloss");
ml_g(x, y) = xgboost(x, 5, label = y);

boost_fit, boost_data, psi_a, psi_b = IRM_Boost(x, y, d, ml_g, ml_m, 3, 0.01);
boost_fit

formula_iivm = @formula(net_tfa ~ p401 + e401 + poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize, 2) + marr + twoearn + db + pira + hown);
formula_iivm = apply_schema(formula_iivm, schema(data));
y, x = modelcols(formula_iivm, data);
d = Integer.(x[:, 2]);
z = Integer.(x[:, 1]);
x = x[:, Not([1, 2])];
size(x, 2)

function IIVM_Lasso(x, y, d, z, ml_g, ml_r, ml_m, nfold, trimming_threshold = 1e-12)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y0_hat = ones(nobser)
    y1_hat = ones(nobser)
    d0_hat = zeros(nobser)
    d1_hat = ones(nobser)
    z_hat = ones(nobser)
    
    dl = convert(Matrix{Float64}, [(d .< 0.5) (d .>= 0.5)])
    zl = convert(Matrix{Float64}, [(z .< 0.5) (z .>= 0.5)])
    
    # Apply learners to y_0, y_1, d_1, d_2 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), z[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        r1_hat = ml_r(x[smp_1, :], dl[smp_1, :])
        m_hat = ml_m(x[foldid[i], :], zl[foldid[i], :])
        
        # Predict: g0_hat, g1_hat, m_hat
        d1_hat[Not(foldid[i])] = GLMNet.predict(r1_hat, x[Not(foldid[i]), :], outtype = :prob)
        y0_hat[Not(foldid[i])] = GLMNet.predict(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = GLMNet.predict(g1_hat, x[Not(foldid[i]), :])
        z_hat[Not(foldid[i])] = GLMNet.predict(m_hat, x[Not(foldid[i]), :], outtype = :prob)
    
    end
    
    
    # Residuals: u0_hat, u1_hat, w0_hat, and w1_hat; no need for residual in z
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    w0_hat = d .- d0_hat
    w1_hat = d .- d1_hat
    
    # Compute regression terms:
    # Left side: y1_hat - y0_hat + z * (u1_hat) / z_hat - (1 - z) * u0_hat / (1 - z_hat)
    psi_b = y1_hat .- y0_hat .+ z .* u1_hat ./ z_hat .- (1 .- z) .* u0_hat ./ (1 .- z_hat)
    
    # Right side: d1_hat - d0_hat + z * (w1_hat) / z_hat - (1 - z) * w0_hat / (1 - z_hat)
    psi_a = d1_hat .- d0_hat .+ z .* w1_hat ./ z_hat .- (1 .- z) .* w0_hat ./ (1 .- z_hat)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, reshape(psi_a, nobser, 1), psi_b)
    
    # Generate data matrix for output
    u_hat = z .* u1_hat + (1 .- z) .* u0_hat
    d_til = z .* w1_hat + (1 .- z) .* w0_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data, psi_a, psi_b;
    
end

Random.seed!(123)

ml_r(x, d) = glmnetcv(x, d, nfolds = 5, Binomial());
ml_g(x, y) = glmnetcv(x, y, nfolds = 5);
ml_m(x, z) = glmnetcv(x, z, nfolds = 5, Binomial());

boost_fit, boost_data, psi_a, psi_b = IIVM_Boost(x, y, d, z, ml_g, ml_r, ml_m, 3)
boost_fit

function IIVM_Forest(x, y, d, z, ml_g, ml_r, ml_m, nfold, trimming_threshold = 1e-12)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y0_hat = ones(nobser)
    y1_hat = ones(nobser)
    d0_hat = zeros(nobser)
    d1_hat = ones(nobser)
    z_hat = ones(nobser)
    
    # Apply learners to y_0, y_1, d_1, d_2 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), z[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        r1_hat = ml_r(x[smp_1, :], d[smp_1])
        m_hat = ml_m(x[foldid[i], :], z[foldid[i]])
        
        # Predict: g0_hat, g1_hat, m_hat
        d1_hat[Not(foldid[i])] = apply_forest(r1_hat, x[Not(foldid[i]), :])
        y0_hat[Not(foldid[i])] = apply_forest(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = apply_forest(g1_hat, x[Not(foldid[i]), :])
        z_hat[Not(foldid[i])] = apply_forest(m_hat, x[Not(foldid[i]), :])
    
    end
    
    
    # Residuals: u0_hat, u1_hat, w0_hat, and w1_hat; no need for residual in z
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    w0_hat = d .- d0_hat
    w1_hat = d .- d1_hat
    
    # Compute regression terms:
    # Left side: y1_hat - y0_hat + z * (u1_hat) / z_hat - (1 - z) * u0_hat / (1 - z_hat)
    psi_b = y1_hat .- y0_hat .+ z .* u1_hat ./ z_hat .- (1 .- z) .* u0_hat ./ (1 .- z_hat)
    
    # Right side: d1_hat - d0_hat + z * (w1_hat) / z_hat - (1 - z) * w0_hat / (1 - z_hat)
    psi_a = d1_hat .- d0_hat .+ z .* w1_hat ./ z_hat .- (1 .- z) .* w0_hat ./ (1 .- z_hat)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, reshape(psi_a, nobser, 1), psi_b)
    
    # Generate data matrix for output
    u_hat = z .* u1_hat + (1 .- z) .* u0_hat
    d_til = z .* w1_hat + (1 .- z) .* w0_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data, psi_a, psi_b;
    
end

Random.seed!(123)

ml_r(x, d) = build_forest(d, x);
ml_g(x, y) = build_forest(y, x);
ml_m(x, z) = build_forest(z, x);

boost_fit, boost_data, psi_a, psi_b = IIVM_Forest(x, y, d, z, ml_g, ml_r, ml_m, 3)
boost_fit

function IIVM_Tree(x, y, d, z, ml_g, ml_r, ml_m, nfold, trimming_threshold = 1e-12)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y0_hat = ones(nobser)
    y1_hat = ones(nobser)
    d0_hat = zeros(nobser)
    d1_hat = ones(nobser)
    z_hat = ones(nobser)
    
    # Apply learners to y_0, y_1, d_1, d_2 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), z[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        r1_hat = ml_r(x[smp_1, :], d[smp_1])
        m_hat = ml_m(x[foldid[i], :], z[foldid[i]])
        
        # Predict: g0_hat, g1_hat, m_hat
        d1_hat[Not(foldid[i])] = apply_tree(r1_hat, x[Not(foldid[i]), :])
        y0_hat[Not(foldid[i])] = apply_tree(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = apply_tree(g1_hat, x[Not(foldid[i]), :])
        z_hat[Not(foldid[i])] = apply_tree(m_hat, x[Not(foldid[i]), :])
    
    end
    
    
    # Residuals: u0_hat, u1_hat, w0_hat, and w1_hat; no need for residual in z
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    w0_hat = d .- d0_hat
    w1_hat = d .- d1_hat
    
    # Compute regression terms:
    # Left side: y1_hat - y0_hat + z * (u1_hat) / z_hat - (1 - z) * u0_hat / (1 - z_hat)
    psi_b = y1_hat .- y0_hat .+ z .* u1_hat ./ z_hat .- (1 .- z) .* u0_hat ./ (1 .- z_hat)
    
    # Right side: d1_hat - d0_hat + z * (w1_hat) / z_hat - (1 - z) * w0_hat / (1 - z_hat)
    psi_a = d1_hat .- d0_hat .+ z .* w1_hat ./ z_hat .- (1 .- z) .* w0_hat ./ (1 .- z_hat)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, reshape(psi_a, nobser, 1), psi_b)
    
    # Generate data matrix for output
    u_hat = z .* u1_hat + (1 .- z) .* u0_hat
    d_til = z .* w1_hat + (1 .- z) .* w0_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data, psi_a, psi_b;
    
end

Random.seed!(123)

ml_r(x, d) = build_tree(d, x, 0, 30, 7, 20, 0.01);
ml_g(x, y) = build_tree(y, x, 0, 30, 7, 20, 0.01);
ml_m(x, z) = build_tree(z, x, 0, 30, 7, 20, 0.01);

boost_fit, boost_data, psi_a, psi_b = IIVM_Tree(x, y, d, z, ml_g, ml_r, ml_m, 3)
boost_fit

psi_b



function IIVM_Boost(x, y, d, z, ml_g, ml_r, ml_m, nfold, trimming_threshold = 1e-12)
    
    # Sample size
    nobser = size(x, 1)
    
    # Fold indexes
    foldid = collect(Kfold(size(x)[1], nfold))
    
    # Initialize vectors for predictions
    y0_hat = ones(nobser)
    y1_hat = ones(nobser)
    d0_hat = zeros(nobser)
    d1_hat = ones(nobser)
    z_hat = ones(nobser)
    
    # Apply learners to y_0, y_1, d_1, d_2 and d separately
    for i in 1:nfold
        # Create y_0 and y_1 for this fold
        mask = findall(==(1), z[foldid[i]])
        smp_1 = foldid[i][mask]
        smp_0 = foldid[i][Not(mask)]
        
        # Model Learning
        g0_hat = ml_g(x[smp_0, :], y[smp_0])
        g1_hat = ml_g(x[smp_1, :], y[smp_1])
        r1_hat = ml_r(x[smp_1, :], d[smp_1])
        m_hat = ml_m(x[foldid[i], :], z[foldid[i]])
        
        # Predict: g0_hat, g1_hat, m_hat
        d1_hat[Not(foldid[i])] = XGBoost.predict(r1_hat, x[Not(foldid[i]), :])
        y0_hat[Not(foldid[i])] = XGBoost.predict(g0_hat, x[Not(foldid[i]), :])
        y1_hat[Not(foldid[i])] = XGBoost.predict(g1_hat, x[Not(foldid[i]), :])
        z_hat[Not(foldid[i])] = XGBoost.predict(m_hat, x[Not(foldid[i]), :])
    
    end
    
    
    # Residuals: u0_hat, u1_hat, w0_hat, and w1_hat; no need for residual in z
    u0_hat = y .- y0_hat
    u1_hat = y .- y1_hat
    w0_hat = d .- d0_hat
    w1_hat = d .- d1_hat
    
    # Compute regression terms:
    # Left side: y1_hat - y0_hat + z * (u1_hat) / z_hat - (1 - z) * u0_hat / (1 - z_hat)
    psi_b = y1_hat .- y0_hat .+ z .* u1_hat ./ z_hat .- (1 .- z) .* u0_hat ./ (1 .- z_hat)
    
    # Right side: d1_hat - d0_hat + z * (w1_hat) / z_hat - (1 - z) * w0_hat / (1 - z_hat)
    psi_a = d1_hat .- d0_hat .+ z .* w1_hat ./ z_hat .- (1 .- z) .* w0_hat ./ (1 .- z_hat)
    
    # Regression with fit(LinearModel, ...)
    rfit = fit(LinearModel, reshape(psi_a, nobser, 1), psi_b)
    
    # Generate data matrix for output
    u_hat = z .* u1_hat + (1 .- z) .* u0_hat
    d_til = z .* w1_hat + (1 .- z) .* w0_hat
    data = DataFrame(u_hat = u_hat, d_til = d_til)
    
    # Function outputs residual data and ATE
    return rfit, data, psi_a, psi_b;
    
end

Random.seed!(123)

ml_r(x, d) = xgboost(x, 5, label = d, objective = "binary:logistic", eval_metric = "logloss");
ml_g(x, y) = xgboost(x, 5, label = y);
ml_m(x, z) = xgboost(x, 5, label = z, objective = "binary:logistic", eval_metric = "logloss");

boost_fit, boost_data, psi_a, psi_b = IIVM_Boost(x, y, d, z, ml_g, ml_r, ml_m, 3)
boost_fit

model = DecisionTreeClassifier(max_depth=2)
DecisionTree.fit!(model, x, d)

predict_proba(model, x)[:, 2]

DecisionTree.predict(model, x)

print_tree(model)
