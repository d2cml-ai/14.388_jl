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
using Statistics, Plots, FixedEffectModels, MLDataUtils, MLBase, RData, Downloads

url = "https://github.com/d2cml-ai/14.388_jl/raw/github_data/data/gun_clean.RData"
download(url, "data.RData")
rdata_read = load("data.RData")
rm("data.RData")
data = rdata_read["data"]
names(data)
println("Number of rows: ",size(data,1))
println("Number of columns: ",size(data,2))

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

using Flux
using Flux: crossentropy, @epochs
using Flux.Data: DataLoader
using Flux: throttle
using Flux: onehotbatch, onecold, @epochs
using StatsBase

Z

mean_1 = mean.(eachcol(z))


std_1 = std.(eachcol(z))



for i in 1:size(z)[2]
    p = (z[:, i] .- mean_1[i]) / std_1[i]
    #colname = names(Z)[i]
    df[!,i] = p
end
    
  

df[:,1] = p

p = (z[:,1].-mean_1[1]) / std_1[1]

[names(Z) mean_1]

function DML2_for_NN(z , d , y, nfold, clu, num_epochs)
    
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
        ##############################################
        ################| MODEL D |###################
        model_y= Chain(Dense(size(z,2), 16, relu), 
        Dense(16, 16, relu),
        Dense(16, 1))

        opt = RMSProp()
        loss_y(x, y) = Flux.Losses.mse(model_y(x), y)
        metrics_y(x, y) = Flux.mae(model_y(x), y)
        ps_y = Flux.params(model_y)

        ##############################################
        ################| MODEL Y |###################
        model_d= Chain(Dense(size(z,2), 16, relu), 
        Dense(16, 16, relu),
        Dense(16, 1))

        opt = RMSProp()
        loss_d(x, y) = Flux.Losses.mse(model_d(x), y)
        metrics_d(x, y) = Flux.mae(model_d(x), y)
        ps_d = Flux.params(model_d)

        data_d = DataLoader((z[foldid[i],:]', d[foldid[i]]'))
        data_y = DataLoader((z[foldid[i],:]', y[foldid[i]]'))

    # Lasso regression, excluding folds selected 
    for epoch in 1:num_epochs
        time = @elapsed Flux.train!(loss_y, ps_y, data_y, opt)
    end

    for epoch in 1:num_epochs
        time = @elapsed Flux.train!(loss_d, ps_d, data_d, opt)
    end

    # Predict estimates using the 
    yhat = model_y(z[Not(foldid[i]),:]')';
    ###############################################################################
    dhat = model_d(z[Not(foldid[i]),:]')';
    
        
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
    
    #return rfit, data;
    
end

# Treatment variable
D = rdata[!,d]

# Outcome variable

Y = rdata[!,y];

# Construct matrix Z
Z = rdata[!, varlis[3:end]];


# Create main variables
z = Matrix(Z);
d = D[!,1];
y = Y[!,1];
clu = rdata[!, :CountyCode];
first(DataFrame(logghomr = y,logfssl = d,CountyCode = clu ),6)

##
DML2_for_NN(z,d,y,10,clu,100)
