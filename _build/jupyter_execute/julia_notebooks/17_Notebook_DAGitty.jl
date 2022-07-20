#import Pkg; Pkg.add("StructuralCausalModels")
#import Pkg; Pkg.add("TikzGraphs")
#import Pkg; Pkg.add("TikzPictures")
#import Pkg; Pkg.add("GraphViz")
#import Pkg; Pkg.add("Dagitty")
#import Pkg; Pkg.add("GraphPlot")

#using StructuralCausalModels
#using TikzGraphs
#using TikzPictures
#using Dagitty, Test 
#using LightGraphs, GraphPlot
#import Pkg; Pkg.add("lav_parse_model_string")

]build GR

Pkg.rm("Plots")
GC.gc()
Pkg.add("Plots")

using Plots

using StructuralCausalModels
using GraphRecipes, Plots
using Random

G = "dag {Z1 -> {X1}; X1 -> {D}; Z1 -> {X2}; Z2 -> X3; X3 -> {Y}; Z2 -> {X2}; D -> {Y}
                ; X2 -> {Y}; X2 -> {D}; M -> {Y}; D -> {M}}"

G = DAG("Model_1", G);

to_ggm(G) |> display

show(G) # Variable's location in an array 

# variables's name into two arrays

names(G.e)

# DAG to matrix
G.e

# colums : affected varaible 
# rows : covariables 

# First: all varibles in (0,0) coordinates

graphplot(G.e, names=names(G.e, 1), curvature_scalar=0, nodesize=0.2,
  method=:spring, fontsize=8, arrow=1, nodeshape=:circle, nodecolor = 2, dim = 2, x = [0,0,0,0,0,0,0,0], y = [0,0,0,0,0,0,0,0])

# curvature_scalar: size point
# arrow : arrow size
# fontsize: circle's size
# (x, y) location 

names(G.e,1)

# X3 in (1,0) coordinates
graphplot(G.e, names=names(G.e, 1), curvature_scalar=0, nodesize=0.2,
  method=:spring, fontsize=8, arrow=1, nodeshape=:circle, nodecolor = 2, dim = 2, x = [0,0,0,0,1,0,0,0], y = [0,0,0,0,0,0,0,0])


names(G.e,1)

# more details
# edgecolor: arrow's color
# axis_buffer: graph's size

graphplot(G.e, names=names(G.e, 1), curvature_scalar=0, nodesize=0.3,
      method=:spring, fontsize=10, arrow=0.2, nodeshape=:circle, nodecolor = :gray,axis_buffer = 0.1
        ,edgecolor = :black, x = [0,-1,1,0,1,-1,1,-1], y = [1,1,1,0,0,0,-1,-1], nodestrokecolor = :black)


G.e

# rows : childer
# columns: parents 


#Parents function

function parents(DAG,x)
    M = DAG.e
    n = size(M)[1]
    v =[]
        for i in 1:n
            B = convert(Int64, M[i,x] == 1)
        if B == 1
           push!(v, names(M,1)[i])
        end
        end
return v
end
 
# Children function

function children(DAG,x)
    M = DAG.e
    n = size(M)[1]
    v =[]
        for i in 1:n
           B = convert(Int64, M[x,i] == 1)
        if B == 1
            push!(v, names(M,1)[i])
        end
        end
return v
end

# Ancentors function

function ancestors(DAG,x)
    CC =[]
    A =  parents(DAG,x)
    push!(CC, [A,x])
    for i in 1:size(A)[1]
        push!(CC, parents(DAG,A[i]))
    end
return CC
end 


# descendants function

function descendants(DAG,x)
    CC =[]
    A =  children(DAG,x)
    push!(CC, [A,x])
    for i in 1:size(A)[1]
        push!(CC, children(DAG,A[i]))
    end
return CC
end 


#Details 

#Declare variables at the beginning makes them as global variables

parents(G,:X2)

ancestors(G,:X2)

children(G,:D)

descendants(G,:X2)

pths = all_paths(G, :D, :Y)
#pths |> display

#Conditional independency between two varaibles given conditioned set of variables

CInd = basis_set(G)
display(CInd)

pths = all_paths(G, :D, :Y)

bp = backdoor_paths(G, pths, :D)

adjustmentsets = adjustment_sets(G, :D, :Y)

e = d_separation(G, :D, :Y)
println("d_separation($(G.name), D, Y) = $e\n")

SWIG = "dag {Z1 -> {X1}; X1 -> {D}; Z1 -> {X2}; Z2 -> X3; X3 -> {Yd}; Z2 -> {X2}; X2 -> {Yd} 
        ; X2 -> {D}; Md -> {Yd}; d -> {Md}}"

SWIG = DAG("Model_2", SWIG)

names(SWIG.e,1)

graphplot(SWIG.e, names=names(SWIG.e, 1), curvature_scalar=0, nodesize=0.3,
      method=:spring, fontsize=10, arrow=0.2, nodeshape=:circle, nodecolor = :gray,axis_buffer = 0.1
        ,edgecolor = :black, x = [0,-0.5,1,0,1,-1,-1,1,-1], y = [1,1,1,0,0,1,0,-1,-1], nodestrokecolor = :black)


#Conditional independency between two varaibles given conditioned set of variables

CInd = basis_set(SWIG)
#length(CInd)


function conditioning_iden(DAG)
n = size(names(DAG.e,1),1)

    for i in 1:n
        a1 = names(DAG.e,1)[i]
        a2 = children(DAG, a1)
        for j in 1:size(a2,1)  
            a3 = adjustment_sets(DAG, a1, a2[j])
            if size(a3)[1] > 0
                println("The effect:", a1, " -> ", a2[j],"  is identifiable by controlling for : " , a3)

            end
        end
    end

end

conditioning_iden(G)

G3 = "dag {D -> {Y}; X -> {D}; X -> {Y}}"

G3 = DAG("Model_3", G3)

names(G3.e,1)

graphplot(G3.e, names=names(G3.e, 1), curvature_scalar=0, nodesize=0.3,
      method=:spring, fontsize=10, arrow=0.2, nodeshape=:circle, nodecolor = :gray,axis_buffer = 0.1
        ,edgecolor = :black, x = [0,-2,0.5], y = [-1,1,2], nodestrokecolor = :black)


#Conditional Independencies
CInd = basis_set(G3) # nothing


bp1 = backdoor_paths(G3, all_paths(G3, :D, :Y), :D) # controling X (confounder varaible)

conditioning_iden(G3)

Random.seed!(1)

to_ggm(G)


