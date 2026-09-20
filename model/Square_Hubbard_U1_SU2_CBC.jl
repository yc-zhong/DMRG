###### module file for Hubbard model on square lattice with NN and NNN interaction

## the file name should be like lattice_model_symmetry_boudary-conditions.
using TensorKit
include("../MPS/TN.jl")

## get the paremeter you used, should be align with you inputs in the command line. the default format is [model,L,j1,t2,j2] and t1=1 by defalut.

const parameter=Dict{String,Any}("model"=>ENV["model"],"L"=>[parse(Int64,ENV["L1"]),parse(Int64,ENV["L2"])],"U"=>parse(Float64,ENV["U"]),"t2"=>parse(Float64,ENV["t2"]),"t3"=>parse(Float64,ENV["t3"]),"Ne"=>parse(Int64,ENV["Ne"]),"t1"=>1.0)

## specify the out_put_file and tmp_file

const out_put_file=parameter["model"]*"/MPS"*string([parameter["L"][1],parameter["L"][2],parameter["Ne"],round.(parameter["U"],digits=6),round.(parameter["t2"],digits=6),round.(parameter["t3"],digits=6)])

const tmp_file="TMP_TN/"*out_put_file

## specify the local physical space
const V=ProductSpace(Vect[(Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0)=>1,(2, 0)=>1, (1, 1/2)=>1))

## specify the go in and go out quantum number laber
const V_in=ProductSpace(Vect[(Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0)=>1))
const V_out=ProductSpace(Vect[(Irrep[U₁] ⊠ Irrep[SU₂])]((parameter["Ne"], 0)=>1))

## sepcify the basic operator matrix for the model
const operator=Dict{String,Matrix}("C_up"=>[0.0 0.0 1.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 1.0 0.0 0.0],
                    "C_dagup"=>[0.0 0.0 0.0 0.0;0.0 0.0 0.0 1.0;1.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0],
                    "C_dn"=>[0.0 0.0 0.0 1.0;0.0 0.0 0.0 0.0;0.0 1.0 0.0 0.0;0.0 0.0 0.0 0.0],
                    "C_dagdn"=>[0.0 0.0 0.0 0.0;0.0 0.0 1.0 0.0;0.0 0.0 0.0 0.0;1.0 0.0 0.0 0.0],
                    "F"=>[1.0 0.0 0.0 0.0;0.0 1.0 0.0 0.0;0.0 0.0 -1.0 0.0;0.0 0.0 0.0 -1.0],
                    "Sz"=>[0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.5 0.0;0.0 0.0 0.0 -0.5],
                    "S+"=>[0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 1.0;0.0 0.0 0.0 0.0],
                    "S-"=>[0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 1.0 0.0],
                    "Nupdn"=>[0.0 0.0 0.0 0.0;0.0 1.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0],
                    "Ntol"=>[0.0 0.0 0.0 0.0;0.0 2.0 0.0 0.0;0.0 0.0 1.0 0.0;0.0 0.0 0.0 1.0]
                    )

## specify the interation matrix for the model and convert to data_type that can be indentified by TensorKit

function TensorKit_matrix(type::String;value::Float64=1.0)
    if type=="Hopping"
        @tensor H_hopping[-1,-2,-3,-4]:=operator["C_dagup"][-1,1]*operator["F"][1,-3]*operator["C_up"][-2,-4]+operator["C_dagdn"][-1,-3]*operator["F"][-2,1]*operator["C_dn"][1,-4]-operator["C_up"][-1,1]*operator["F"][1,-3]*operator["C_dagup"][-2,-4]-operator["C_dn"][-1,-3]*operator["F"][-2,1]*operator["C_dagdn"][1,-4]
        H_hopping=reshape(H_hopping,(4,4,4,4))
        return TensorMap(-1.0*H_hopping*value,V'*V',V'*V')
    elseif type=="F"
        return TensorMap(operator["F"],V',V')
    elseif type=="Sz"
        return TensorMap(operator["Sz"]*value,V',V')
    elseif type=="Ntol"
        return TensorMap(operator["Ntol"]*value,V',V')
    elseif type=="Nupdn"
        return TensorMap(operator["Nupdn"]*value,V',V')
    end
end

## define the 2D to 1D Mapping for MPS. Snake by default
function Mapping(po::Vector{Int64})
    L=parameter["L"]
    MAP=[[i for i in 1: L[2]] for j in 1:L[1]]
    for j in 1:L[2]
        for i in 1:L[1]
            MAP[i][j]=(j-1)*L[1]+i
        end
    end
    return MAP[po[1]][po[2]]
end

function split_Ham(a::TensorMap,p1,p2;interval_dict=Dict(1=>1,2=>1,3=>1,4=>1,5=>1,6=>1))
    L,t,R,_=tsvd(a,p1,p2;trunc=truncbelow(1e-5))
    tt=Vector{Vector{TensorMap}}()
    #println(domain(t))
    #out=TensorMap(zeros,eltype(a),codomain(a),domain(a))
    for c in blocksectors(domain(t))
        I = sectortype(t)
        S = spacetype(t)
        A = storagetype(a)
        LAdata = Dict{I, A}()
        LBdata = Dict{I, A}()
        LCdata = Dict{I, A}()
        LAdata[c]=block(L,c)
        LBdata[c]=block(R,c)
        LCdata[c]=block(t,c)
        sector_dims = Dict{I, Int}()
        sector_dims[c] = size(LCdata[c], 2)
        if size(LCdata[c],2)>=2
            st=0
            ed=1
            test_dim=Dict{I, Int}()
            test_dim[c]=1
            interval=interval_dict[dim(S(test_dim))]
            while ed<size(LCdata[c], 2)
                ed=st+interval
                if abs(ed-size(LCdata[c], 2))<1 || ed>size(LCdata[c], 2)
                    ed=size(LCdata[c], 2)
                end
                LA_small = Dict{I, A}()
                LB_small = Dict{I, A}()
                LC_small = Dict{I, A}()
                LA_small[c]=LAdata[c][:,st+1:ed]
                LC_small[c]=LCdata[c][st+1:ed,st+1:ed]
                LB_small[c]=LBdata[c][st+1:ed,:]
                sector_dims_small = Dict{I, Int}()
                sector_dims_small[c] = size(LC_small[c], 2)
                st=ed
                V = S(sector_dims_small)
                W = ProductSpace(V)
                tt=vcat(tt,[[TensorMap(LA_small, codomain(L),W)*TensorMap(LC_small, W,W),TensorMap(LB_small,W,domain(R))]])
            end
        else
            V = S(sector_dims)
            W = ProductSpace(V)
            tt=vcat(tt,[[TensorMap(LAdata, codomain(L),W)*TensorMap(LCdata, W,W),TensorMap(LBdata,W,domain(R))]])
        end
    end
    return tt
end

H_Hopping=split_Ham(TensorKit_matrix("Hopping"),(1,3),(2,4))

function Ham()
    Ham_matrix=Dict{Int64,Vector{TensorMap}}()
    for i in 1:length(H_Hopping)
        Ham_matrix[-i]=[H_Hopping[i][1],H_Hopping[i][2]]
    end
    return Ham_matrix
end

const Ham_matrix=Ham()

## specify all the hamiltonian terms

function Hamiltonian_terms()
    terms = Dict{Vector{Int64}, String}()
    Lx, Ly = parameter["L"][1], parameter["L"][2]
    for i in 1:Lx, j in 1:Ly
        k0 = [i, j]
        # NN: right (OBC in y)
        k1 = j < Ly ? [i, j+1] : nothing
        # NN: up (PBC in x)
        k2 = [i < Lx ? i+1 : 1, j]
        # NNN: up-right
        k3 = (j < Ly) ? [i < Lx ? i+1 : 1, j+1] : nothing
        # NNN: up-left
        k4 = (j > 1) ? [i < Lx ? i+1 : 1, j-1] : nothing
        # NNNN: up-up (PBC)
        k5 = (Lx > 2) ? [i <= Lx-2 ? i+2 : i-Lx+2, j] : nothing
        # NNNN: right-right (OBC)
        k6 = (Ly > 2 && j < Ly-1) ? [i, j+2] : nothing

        for m in 1:length(H_Hopping)
            # NN
            if k1 !== nothing
                terms[vcat(sort([Mapping(k0), Mapping(k1)]), [-m])] = "t1"
            end
            terms[vcat(sort([Mapping(k0), Mapping(k2)]), [-m])] = "t1"
            # NNN
            if parameter["t2"] != 0
                if k3 !== nothing
                    terms[vcat(sort([Mapping(k0), Mapping(k3)]), [-m])] = "t2"
                end
                if k4 !== nothing
                    terms[vcat(sort([Mapping(k0), Mapping(k4)]), [-m])] = "t2"
                end
            end
            # NNNN
            if parameter["t3"] != 0
                if k5 !== nothing
                    keys = vcat(sort([Mapping(k0), Mapping(k5)]), [-m])
                    if !haskey(terms, keys)
                        terms[keys] = "t3"
                    end
                    # terms[vcat(sort([Mapping(k0), Mapping(k5)]), [-m])] = "t3"
                end
                if k6 !== nothing
                    terms[vcat(sort([Mapping(k0), Mapping(k6)]), [-m])] = "t3"
                end
            end
        end
    end
    return terms
end

function Hamiltonian_terms_onsite()
    terms_onsite=Dict{Int64,TensorMap}()
    L=parameter["L"]
    for i in 1:L[1]
        for j in 1:L[2]
            k0=[i,j]
            terms_onsite[Mapping(k0)]=TensorKit_matrix("Nupdn",value=parameter["U"])
        end
    end
    return terms_onsite
end

const terms_onsite=Hamiltonian_terms_onsite()
const terms=Hamiltonian_terms()

## define the initial MPS function

# function TN_initial()
#     L_num=parameter["L"][1]*parameter["L"][2]
#     temp_t=TensorMap(zeros,Float64,V',V')
#     doping=1-parameter["Ne"]/L_num
#     tensors=Vector{TensorMap}(undef, L_num)
#     previous_space=V
#     for i in 1:L_num
#         second_space=blocksectors(previous_space*V)
#         I = sectortype(temp_t)
#         S = spacetype(temp_t)
#         sector_dims = Dict{I, Int}()
#         count=0
#         for k in second_space
#             kk=k.sectors[1].charge
#             kj=k.sectors[2].charge
#             ii=ceil(i*(1-doping))
#             if abs(kk-ii-1)<=3
#                 if kj-1<=0
#                     sector_dims[k]=1
#                 end
#             end
#         end
#         tensors[i]=TensorMap(randn,Float64, previous_space*V,S(sector_dims))
#         if (i%3==4)
#             tensors[i]=TensorMap(ones,previous_space*V,S(sector_dims))
#         end
#         if i == L_num
#             tensors[i]=TensorMap(randn,Float64, previous_space*V,V_out)
#         end
#         if i==1
#             tensors[i]=TensorMap(randn,Float64, V_in*V,V)
#         end
#         if i>1
#             previous_space=S(sector_dims)
#         end
#     end
#     return tensors
# end

# function Sweep(step::Int64)
#     return sweep()
# end

function map_initial(L::Vector{Int64})
    map=[0 for i in 1:L[2]*L[1]]
    #1/8 hole doping
    for i in 1:L[1]
        for j in 1:L[2]     
            if  j%8 in [1,2,3,4]
                map[Mapping([i,j])]=(-1)^(i+j+1)
            elseif j%8 in [0,6,7]
                map[Mapping([i,j])]=(-1)^(i+j)
            end
        end
    end
    map1=[abs(map[i]) for i in 1:L[2]*L[1]]
    return map,map1
end

function TN_initial()
    L_num=parameter["L"][1]*parameter["L"][2]
    map,map1=map_initial(parameter["L"])
    tensors=Vector{TensorMap}(undef, L_num)
    previous_space=V_in
    for i in 1:L_num
        N_SU2=sum(map1[1:i])%2
        after_space=ProductSpace(Vect[(Irrep[U₁] ⊠ Irrep[SU₂])]((sum(map1[1:i]), N_SU2/2)=>1))
        tmp_tensor=isometry(Matrix{Float64},previous_space*V,after_space)
        tensors[i]=tmp_tensor
        previous_space=after_space
    end
    return tensors
end
