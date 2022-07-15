import Pkg


## Install package 

#import Pkg; Pkg.add("CairoMakie")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("Distributions")
#import Pkg; Pkg.add("HypothesisTests")
#Pkg.add("ClinicalTrialUtilities")
using CairoMakie  # for density plot 
using ClinicalTrialUtilities
using Random   # for random seeds
using Statistics 
using Distributions
using HypothesisTests
using Images, FileIO

NV = 200745 # number of vaccinated (treated)
NU = 201229 # number of unvaccinated (control)
RV= 33/NV  # average outcome for vaccinated
RU =115/NU  # average outcome for unvaccinated
VE = (RU - RV)/RU # vaccine efficacy

## Incidence per 100000

n = 100000

IncidenceRV = RV*n
IncidenceRU = RU*n

#Treatment effect - estimated reduction in incidence of polio per 100000 people

ATE_hat = n*(RV-RU)

println("Incidence per 100000 among treated: ", round(IncidenceRV; digits =4))

println("Incidence per 100000 among untreated: ", round(IncidenceRU; digits =4))

println("Estimated ATE of occurances per 100,000 is: ", round(ATE_hat; digits =4))

# outcomes (RV, RU) are Bernoulli
# variance, standard deviation and confidence interval of ATE 

var_RV = RV*(1-RV)/NV
var_RU = RU*(1-RU)/NU

var_ATE_hat  = n^2*(var_RV+var_RU)

std_ATE_hat = sqrt(var_ATE_hat)

println("Standard deviation for ATE is:  ", round(std_ATE_hat; digits =4))



# Confidence interval 

CI_ATE_hat =[ round(ATE_hat - 1.96*std_ATE_hat; digits = 4), round(ATE_hat + 1.96*std_ATE_hat; digits = 4)]

println("95% confidence interval of ATE is $CI_ATE_hat")

println("Overall VE is: " , round(VE; digits =4))

# Confidence interval based on approximate bootstrap

# Monte Carlo draws

Random.seed!(1)

B = 10000 #  number of boostraps
RVs = RV*ones(Float16, B)  + randn(B)*sqrt(var_RV)
RUs = RU*ones(Float16, B)  + randn(B)*sqrt(var_RU)
VEs= (RUs - RVs)./RUs


## Confidence interval of VEs

CI_VE = [round(quantile!(VEs, 0.025); digits = 4 ), round(quantile!(VEs, 0.975); digits = 4)]

println("two-sided 95% confidence interval of VE is $CI_VE")

## Approximate distribution of VEs estimates 

f = Figure(resolution = (600, 400))
Axis(f[1,1], title = "Approximate distribution of VEs estimates ")
    
density!(VEs, color = (:red, 0.1), strokecolor = :red, strokewidth = 1, strokearound = true, bandwidth = 0.008257) 

f


img_path = "../data/imagen_RCT.png"
img = load(img_path)
imresize(img, ratio=1)

# define inputs 

NV =  19965; # number vaccinated
NU =  20172; # number unvaccinated
RV = 9/NV; # average outcome for vaccinated
RU = 169/NU; # average outcome for unvaccinated
VE = (RU - RV)/RU; # vaccine efficacy


## Incidence per 100000

n = 100000

IncidenceRV = RV*n
IncidenceRU = RU*n

#Treatment effect - estimated reduction in incidence of polio per 100000 people

ATE_hat = n*(RV-RU)

println("Incidence per 100000 among treated: ", round(IncidenceRV; digits =4))

println("Incidence per 100000 among untreated: ", round(IncidenceRU; digits =4))

println("Estimated ATE of occurances per 100 000 is: ", round(ATE_hat; digits =4))

# outcomes (RV, RU) are Bernoulli
# variance, standard deviation and confidence interval of ATE 

var_RV = RV*(1-RV)/NV
var_RU = RU*(1-RU)/NU

var_ATE_hat  = n^2*(var_RV+var_RU)

std_ATE_hat = sqrt(var_ATE_hat)

println("Standard deviation for ATE is:  ", round(std_ATE_hat; digits =4))


# Confidence interval 

CI_ATE_hat =[ round(ATE_hat - 1.96*std_ATE_hat; digits = 4), round(ATE_hat + 1.96*std_ATE_hat; digits = 4)]

println("95% confidence interval of ATE is $CI_ATE_hat")

println("Overall VE is: " , round(VE; digits =4))

Random.seed!(1)

B = 10000 #  number of boostraps
RVs = RV*ones(Float16, B)  + randn(B)*sqrt(var_RV)
RUs = RU*ones(Float16, B)  + randn(B)*sqrt(var_RU)
VEs= (RUs - RVs)./RUs


## Confidence interval of VEs

CI_VE = [round(quantile!(VEs, 0.025); digits = 4 ), round(quantile!(VEs, 0.975); digits = 4)]

println("two-sided 95% confidence interval of VE is $CI_VE")

## Approximate distribution of VEs estimates 

f = Figure(resolution = (500, 400))
Axis(f[1,1], title = "Approximate distribution of VEs estimates ")
    
density!(VEs, color = (:blue, 0.1), strokecolor = :blue, strokewidth = 1, strokearound = true) 

f

NV =  3239+805;
NU =  3255+812;
RV = 1/NV;
RU = (14+5)/NU;
VE = (RU - RV)/RU;

println("Overall VE is: " , round(VE; digits =4))

var_RV = RV*(1-RV)/NV
var_RU = RU*(1-RU)/NU

Random.seed!(1)

B = 10000 #  number of boostraps

RVs = RV*ones(Float16, B)  + randn(B)*sqrt(var_RV) + 10^(-10)*ones(Float16, B) 
RUs = RU*ones(Float16, B)  + randn(B)*sqrt(var_RU) + 10^(-10)*ones(Float16, B) 
VEs= (RUs - RVs)./RUs



## Confidence interval of VEs

CI_VE = [round(quantile!(VEs, 0.025); digits = 4 ), round(quantile!(VEs, 0.975); digits = 4)]

OneSidedCI_VE = [round(quantile!(VEs, 0.05); digits = 4 ), 1]

println("two-sided  95% confidence interval of VE is $CI_VE")

println("one side 95% confidence interval of VE is $OneSidedCI_VE")

## Approximate distribution of VEs estimates 

f = Figure(resolution = (500, 400))
Axis(f[1,1], title = "Approximate distribution of VEs estimates ")
    
density!(VEs, color = (:gray, 0.1), strokewidth = 1, strokearound = true) 

f

NV =  3239+805;
NU =  3255+812;
RV = 1/NV;
RU = (14+5)/NU;
VE = (RU - RV)/RU;

println("Overall VE is: " , round(VE; digits =4))

Random.seed!(1)

B = 10000 #  number of boostraps
VEs= (RUs - RVs)./RUs

RVs = rand(Binomial(NV,RV), B)
RUs = rand(Binomial(NU,RU), B)
VEs= (RUs - RVs)./RUs

## Confidence interval of VEs

CI_VE = [round(quantile!(VEs, 0.025); digits = 4 ), round(quantile!(VEs, 0.975); digits = 4)]

OneSidedCI_VE = [round(quantile!(VEs, 0.05); digits = 4 ), 1]

println("two-sided 95% confidence interval of VE is $CI_VE")

println("one sided 95% confidence interval of VE is $OneSidedCI_VE")

## Approximate distribution of VEs estimates 

f = Figure(resolution = (500, 400))
Axis(f[1,1], title = "Approximate distribution of VEs estimates ")
    
density!(VEs, color = (:gray, 0.1), strokewidth = 1, strokearound = true) 

f

# Exact CI exploiting Bernoulli outcome using the Cornfield Procedure

NV =  19965;
NU =  20172;
RV = 9/NV;
RU = 169/NU;
VE = (RU - RV)/RU;

confint = orpropci(9::Int, NV::Int, 169::Int, NU::Int; alpha = 0.05)::ConfInt


l=1-0.027616272602670834
u=1-0.10317864685976501
confint = (l,u)

# Exact CI exploiting Bernoulli outcome for the two groups that are 65 or older

NV =  3239+805;
NU =  3255+812;
RV = 1/NV;
RU = (14+5)/NU;
VE = (RU - RV)/RU;

confint = orpropci(1::Int, NV::Int, 19::Int, NU::Int; alpha = 0.05)::ConfInt


l=1-0.00896277602085047
u=1-0.30983022294266616
confint = (l,u)
