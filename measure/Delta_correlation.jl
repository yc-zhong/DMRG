using MKL
using LinearAlgebra
using TensorOperations
using TensorKit
using ProgressMeter
include("../MPS/TN.jl")
# BLAS.set_num_threads(10)


function Delta(D,out_put_file,L)
    save_file="TNc/"*out_put_file*"_"*string(D)
    L_num=L[1]*L[2]
    mc=0.0
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(1)))<=1.01
            V=space(tensor_load(save_file*"/"*string(1)),2)
            mc1,mc2=delta1(D,out_put_file,L,V)
        end
        if norm(tensor_load(save_file*"/"*string(L_num)))<=1.01
            V=space(tensor_load(save_file*"/"*string(1)),2)
            mc1,mc2=delta2(D,out_put_file,L,V)
        end
        out_put_short_file="out_short/"*out_put_file*"_Delta"
        write_data_txt(out_put_short_file,string(vcat([D],vec(mc1),vec(mc2))))
    end
    return mc1,mc2
end

if !isdefined(@__MODULE__, :HubbardU1U1Operators)
    include("../model/Hubbard_U1_U1_operators.jl")
end
using .HubbardU1U1Operators: hubbard_u1u1_operators

const operator = hubbard_u1u1_operators()

function delta1(D,out_put_file,L,V)
    @tensor H_delta[-1,-2,-3,-4]:=operator["C_up"][-1,1]*operator["F"][1,-3]*operator["F"][-2,2]*operator["C_dn"][2,-4]+operator["C_dn"][-1,-3]*operator["C_up"][-2,-4]+operator["C_dagup"][-1,1]*operator["F"][1,-3]*operator["F"][-2,2]*operator["C_dagdn"][2,-4]+operator["C_dagdn"][-1,-3]*operator["C_dagup"][-2,-4]
    save_file="TNc/"*out_put_file*"_"*string(D)
    ham=TensorMap(H_delta/(2*sqrt(2)),V'*V',V'*V')
    F=TensorMap(operator["F"],V',V')
    ham=compress_t(ham,(1,3),(2,4))
    L_num=L[1]*L[2]
    st=false
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(1)))<=1.01
            st=true
        end
    end
    if st
        m_data_x=zeros(L[1],L[2]) # data for delta
        m_data_y=zeros(L[1],L[2]) # data for delta
        TMP_tensor = Vector{TensorMap}(undef, L[1])
        ttn1=tensor_load(save_file*"/"*string(1))
        p_bar = Progress(L_num,desc="Delta",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
        A=0
        before=0
        for i in 1:L_num
            ttn1=tensor_load(save_file*"/"*string(i))
            ProgressMeter.next!(p_bar)
            k1=(i-1)%L[1]+1
            k2=fld(i-1,L[1])+1
            @tensor B[-1,-2,-3]:=ttn1[-1,3,6]*ham[2][-2,3,5]*conj(ttn1[-3,5,6])
            if k2!=1
                @tensor A[]:=TMP_tensor[k1][1,2,3]*B[1,2,3]
                m_data_y[k1,k2-1]=values(A.data)[1][1]
            end
            if i==1
                @tensor A[-1,-2,-3]:=ttn1[1,2,-1]*ham[1][2,3,-2]*conj(ttn1[1,3,-3])
                TMP_tensor[k1]=A
            else
                @tensor A[-1,-2,-3]:=before[1,2]*ttn1[1,3,-1]*ham[1][3,4,-2]*conj(ttn1[2,4,-3])
                TMP_tensor[k1]=A
            end
            if k1!=1
                @tensor A[]:=TMP_tensor[k1-1][1,2,3]*B[1,2,3]
                m_data_x[k1-1,k2]=values(A.data)[1][1]
                if k1==L[1]
                    @tensor A[]:=TMP_tensor[1][1,2,3]*B[1,2,3]
                    m_data_x[L[1],k2]=values(A.data)[1][1]
                end
            end
            for j in 1:L[1]
                if k2==1
                    if j<k1
                        @tensor A[-1,-2,-3]:=TMP_tensor[j][2,-2,3]*ttn1[2,1,-1]*conj(ttn1[3,4,-3])*F[1,4]
                        TMP_tensor[j]=A
                    end
                else
                    if j!=k1 
                        @tensor A[-1,-2,-3]:=TMP_tensor[j][2,-2,3]*ttn1[2,1,-1]*conj(ttn1[3,4,-3])*F[1,4]
                        TMP_tensor[j]=A
                    end
                end
            end
            if i<L_num
                if i==1
                    @tensor before[-1,-2]:=ttn1[1,2,-1]*conj(ttn1[1,2,-2])
                else
                    @tensor before[-1,-2]:=before[1,2]*ttn1[1,3,-1]*conj(ttn1[2,3,-2])
                end
            end
        end
        #return m_data
        center=fld(L[2],2)
        println("Delta_ALL")
        println(m_data_x)
        println(m_data_y)
        # println(sum(m_data[:,center+1])/L[1])
        return m_data_x, m_data_y
    end
end
