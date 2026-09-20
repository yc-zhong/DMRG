using MKL
using LinearAlgebra
using TensorOperations
using ProgressMeter
include("../MPS/TN.jl")
if !isdefined(@__MODULE__, :HubbardU1U1Operators)
    include("../model/Hubbard_U1_U1_operators.jl")
end
using .HubbardU1U1Operators: hubbard_u1u1_operators

const number_operator = hubbard_u1u1_operators()["Ntol"]
BLAS.set_num_threads(10)

function write_data_txt(file_name::String,data::String)
    open(file_name*".txt", "a+") do file
        write(file, data*"\n")
    end
    return 0
end

function N(D,out_put_file,L)
    save_file="TNc/"*out_put_file*"_"*string(D)
    L_num=L[1]*L[2]
    mc=0.0
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(1)))<=1.01
            V=space(tensor_load(save_file*"/"*string(1)),2)
            mc=N1(D,out_put_file,L,V)
        end
        if norm(tensor_load(save_file*"/"*string(L_num)))<=1.01
            V=space(tensor_load(save_file*"/"*string(1)),2)
            mc=N2(D,out_put_file,L,V)
        end
        out_put_short_file="out_short/"*out_put_file*"_N"
        write_data_txt(out_put_short_file,string(vcat([D,sum(mc)*L[1]],mc)))
    end
    return mc
end

function N1(D,out_put_file,L,V)
    sz = number_operator
    if dim(V)==3
        sz=[0.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
    end
    save_file="TNc/"*out_put_file*"_"*string(D)
    ham=TensorMap(sz,V', V')
    L_num=L[1]*L[2]
    st=false
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(1)))<=1.01
            st=true
        end
    end
    if st
        m_data=zeros(L[1],L[2]) # data for sz
        ttn1=tensor_load(save_file*"/"*string(1))
        @tensor before[-1,-2]:=ttn1[1,2,-1]*conj(ttn1[1,2,-2])
        p_bar = Progress(L_num,desc="M_Z",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
        for i in 1:L_num
            ProgressMeter.next!(p_bar)
            if i%L[1]==0
                k1=L[1]
            else
                k1=i%L[1]
            end
            k2=fld(i-1,L[1])+1
            @tensor A[-1,-2,-3]:=ttn1[-1,1,-3]*ham[1,-2]
            if i==1
                @tensor A[]:=A[1,2,3]*conj(ttn1[1,2,3])
                m_data[k1,k2]=values(A.data)[1][1]
            else
                @tensor A[]:=A[1,2,3]*before[1,2,3]
                m_data[k1,k2]=values(A.data)[1][1]
            end
            if i<L_num
                st=time()
                ttn2=tensor_load(save_file*"/"*string(i+1))
                if i==1
                    @tensor before[-1,-2,-3]:=before[-1,1]*conj(ttn2[1,-2,-3])
                else
                    @tensor before[-1,-2,-3]:=before[1,2,3]*ttn1[1,2,-1]*conj(ttn2[3,-2,-3])
                end
                ttn1=ttn2
            end
        end
        #return m_data
        for i in 1:L[1]
            for j in 1:L[2]
                m_data[i,j]=m_data[i,j]
            end
        end
        println(m_data)
        println("M_Z_Center")
        M_data=[0.0 for i in 1:L[2]]
        for i in 1:L[2]
            M_data[i]=sum(m_data[:,i])/L[1]
        end
        # println(sum(m_data[:,center+1])/L[1])
        return M_data
    end
end

function N2(D,out_put_file,L,V)
    sz = number_operator
    if dim(V)==3
        sz=[0.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
    end
    save_file="TNc/"*out_put_file*"_"*string(D)
    ham=TensorMap(sz,V', V')
    L_num=L[1]*L[2]
    st=false
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(L_num)))<=1.01
            st=true
        end
    end
    if st
        m_data=zeros(L[1],L[2]) # data for sz
        ttn1=tensor_load(save_file*"/"*string(L_num))
        @tensor before[-1,-2]:=ttn1[-1,2,1]*conj(ttn1[-2,2,1])
        p_bar = Progress(L_num,desc="N",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
        for i in L_num:-1:1
            ProgressMeter.next!(p_bar)
            if i%L[1]==0
                k1=L[1]
            else
                k1=i%L[1]
            end
            k2=fld(i-1,L[1])+1
            @tensor A[-1,-2,-3]:=ttn1[-1,1,-3]*ham[1,-2]
            if i==L_num
                @tensor A[]:=A[1,2,3]*conj(ttn1[1,2,3])
                m_data[k1,k2]=values(A.data)[1][1]
            else
                @tensor A[]:=A[1,2,3]*before[1,2,3]
                m_data[k1,k2]=values(A.data)[1][1]
            end
            if i>1
                st=time()
                ttn2=tensor_load(save_file*"/"*string(i-1))
                if i==L_num
                    @tensor before[-1,-2,-3]:=before[-3,1]*conj(ttn2[-1,-2,1])
                else
                    @tensor before[-1,-2,-3]:=before[3,2,1]*ttn1[-3,2,1]*conj(ttn2[-1,-2,3])
                end
                ttn1=ttn2
            end
        end
        #return m_data
        for i in 1:L[1]
            for j in 1:L[2]
                m_data[i,j]=m_data[i,j]
            end
        end
        println(m_data)
        println("M_Z_Center")
        M_data=[0.0 for i in 1:L[2]]
        for i in 1:L[2]
            M_data[i]=sum(m_data[:,i])/L[1]
        end
        # println(sum(m_data[:,center+1])/L[1])
        return M_data
    end
end

# N()
