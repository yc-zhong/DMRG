using LinearAlgebra
using TensorOperations
using ProgressMeter
import FileIO,JLD2

if !isdefined(@__MODULE__, :HubbardU1U1Operators)
    include("../model/Hubbard_U1_U1_operators.jl")
end
using .HubbardU1U1Operators: hubbard_u1u1_operators

const operator = hubbard_u1u1_operators()

function TensorKit_matrix_slt(V=Vect[(Irrep[U₁] ⊠ Irrep[U₁])]((0, 0)=>1,(2, 0)=>1, (1, 1/2)=>1,(1, -1/2)=>1))
    if isfile("measure/"*string(hash(V))*"_Slt.jld2")
        slt=FileIO.load("measure/"*string(hash(V))*"_Slt.jld2")["ttn_tem"]
        TAB,TCD=compress_t(slt,(1,2,5,6),(3,4,7,8),1)
        TA,TB = compress_t(TAB,(1,3),(2,4,5),1)
        TC,TD = compress_t(TCD,(1,2,4),(3,5),1)
        return [TA,TB,TC,TD]
    else
        @tensor s[a,b,c,d,a',b',c',d']:= 
        -operator["C_dagup"][a,a1]*operator["F"][a1,a'] * operator["C_dagdn"][b,b1]*operator["F"][b1,b'] * operator["C_up"][c,c1]*operator["F"][c1,c'] * operator["F"][d,d1]*operator["C_dn"][d1,d']-
        operator["C_dagdn"][a,a'] * operator["C_dagup"][b,b'] * operator["F"][c,c1]*operator["C_dn"][c1,c2]*operator["F"][c2,c'] * operator["C_up"][d,d']+
        operator["C_dagup"][a,a1]*operator["F"][a1,a'] * operator["C_dagdn"][b,b1]*operator["F"][b1,b'] * operator["F"][c,c1]*operator["C_dn"][c1,c2]*operator["F"][c2,c'] * operator["C_up"][d,d']+
        operator["C_dagdn"][a,a'] * operator["C_dagup"][b,b'] * operator["C_up"][c,c1]*operator["F"][c1,c'] * operator["F"][d,d1]*operator["C_dn"][d1,d']  
        slt = TensorMap(s,V'*V'*V'*V',V'*V'*V'*V')
        println("Slt tensor created, saving to file...")
        FileIO.save("measure/"*string(hash(V))*"_Slt.jld2","ttn_tem",slt)
        TAB,TCD=compress_t(slt,(1,2,5,6),(3,4,7,8),1)
        TA,TB = compress_t(TAB,(1,3),(2,4,5),1)
        TC,TD = compress_t(TCD,(1,2,4),(3,5),1)
        return [TA,TB,TC,TD]
    end
end

function Mapping(po::Vector{Int64},L)
    if po[1]>L[1]
        po[1]=po[1]-L[1]
    end
    if po[2]>L[2]
        po[2]=po[2]-L[2]
    end
    
    MAP=[[i for i in 1: L[2]] for j in 1:L[1]]
    for j in 1:L[2]
        for i in 1:L[1]
            MAP[i][j]=(j-1)*L[1]+i
        end
    end
    return MAP[po[1]][po[2]]
end


function CORR_slt(D,out_put_file,L,mode="v")
    save_file="TNc/"*out_put_file*"_"*string(D)
    L_num=L[1]*L[2]
    mc=0.0
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(1)))<=1.01
            V=space(tensor_load(save_file*"/"*string(1)),2)
            mc=CORR_slt1(D,out_put_file,L,V,mode)
        else
            norm_TN2(1,save_file,L=L)
            V=space(tensor_load(save_file*"/"*string(1)),2)
            mc=CORR_slt1(D,out_put_file,L,V,mode)
        end
        out_put_short_file="out_short/"*out_put_file*"_CORR_slt_"*mode
        mc[1]=D
        write_data_txt(out_put_short_file,string(mc))
    end
    return mc
end

function CORR_slt1(D,out_put_file,L,V,mode)
    F=TensorMap(operator["F"],V',V')
    println(V)
    save_file="TNc/"*out_put_file*"_"*string(D)
    mid=fld(L[1],2)
    ref=5
    idx_ = sort([Mapping([mid,ref],L),Mapping([mid+1,ref],L)])  ## reference point
    St=ref+1  ## measure start point
    idx2_=[sort([Mapping([mid,i],L),Mapping([mid+1,i],L)]) for i in St:L[2]]  ## measure point
    if mode == "h"
        # idx_ = [Mapping([mid,ref],L),Mapping([mid,rer+1],L)]  ## reference point
        # St=3   ## measure start point
        idx2_=[sort([Mapping([mid,i],L),Mapping([mid,i+1],L)]) for i in St:L[2]-1]  ## measure point
        St=St+1
    end
    idx2_first = Dict(x[1] => x[2] for x in idx2_)
    Op_=TensorKit_matrix_slt(V)
    A=0
    Corr_data=[0.0 for i in 1:L[2]]
    p_bar = Progress(L[1]*L[2],desc="CORR_slt",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
    for k in 1:L[1]*L[2]
        ProgressMeter.next!(p_bar)
        ttn = tensor_load(save_file*"/"*string(k))
        if k < idx_[1]
            if k == 1
                @tensor A[-1,-2]:=ttn[1,2,-1]*conj(ttn[1,2,-2])
            else
                @tensor A[-1,-2]:=A[1,3]*ttn[1,2,-1]*conj(ttn[3,2,-2])
            end
        elseif k == idx_[1]
            @tensor A[-1,-2,-3]:=A[1,2]*ttn[1,3,-1]*Op_[1][3,4,-2]*conj(ttn[2,4,-3])
        elseif idx_[1] < k < idx_[2]
            @tensor A[-1,-2,-3]:=A[1,-2,3]*ttn[1,2,-1]*F[2,4]*conj(ttn[3,4,-3])
        elseif k == idx_[2]
            @tensor A[-1,-2,-3]:=A[1,2,3]*ttn[1,4,-1]*Op_[2][2,4,5,-2]*conj(ttn[3,5,-3])
        elseif k > idx_[2]
            BB = deepcopy(A)
            @tensor A[-1,-2,-3]:=A[1,-2,3]*ttn[1,2,-1]*conj(ttn[3,2,-3])
            if k in collect(keys(idx2_first))
                @tensor A2[-1,-2,-3] := BB[1,2,3]*ttn[1,4,-1]*Op_[3][2,4,5,-2]*conj(ttn[3,5,-3])
                for kk in k+1:idx2_first[k]-1
                    ttn = tensor_load(save_file*"/"*string(kk))
                    @tensor A2[-1,-2,-3] := A2[1,-2,3]*ttn[1,2,-1]*F[2,4]*conj(ttn[3,4,-3]) 
                end
                ttn = tensor_load(save_file*"/"*string(idx2_first[k]))
                @tensor A2[] := A2[1,2,3]*ttn[1,4,6]*Op_[4][2,4,5]*conj(ttn[3,5,6])
                Corr_data[St]=values(A2.data)[1][1]
                St=St+1
            end
        end
    end
    return Corr_data
end
