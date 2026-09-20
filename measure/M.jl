using MKL
using LinearAlgebra
using TensorOperations
using ProgressMeter
include("../MPS/TN.jl")
BLAS.set_num_threads(10)

function M(D,out_put_file,L)
    save_file="TNc/"*out_put_file*"_"*string(D)
    L_num=L[1]*L[2]
    mc=0.0
    if isdir(save_file)
        if norm(tensor_load(save_file*"/"*string(1)))<=1.01
            mc=M1(D,out_put_file,L)
        end
        if norm(tensor_load(save_file*"/"*string(L_num)))<=1.01
            mc=M2(D,out_put_file,L)
        end
        out_put_short_file="out_short/"*out_put_file*"_Mc"
        write_data_txt(out_put_short_file,string(vcat([D],mc)))
    end
    return mc
end

function M1(D,out_put_file,L)
    sz=[0.5 0.0; 0.0 -0.5]
    save_file="TNc/"*out_put_file*"_"*string(D)
    V=U₁Space(-1/2=>1,1/2=>1)
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
        center=fld(L[2],2)
        println("M_Z_ALL")
        println(m_data)
        println("M_Z_Center")
        println(sum(abs.(m_data[:,center+1]))/L[1])
        return [sum(abs.(m_data[1:2:end,center+1]))/L[1]*2, sum(abs.(m_data[2:2:end,center+1]))/L[1]*2]
    end
end


function M2(D,out_put_file,L)
    sz=[0.5 0.0; 0.0 -0.5]
    save_file="TNc/"*out_put_file*"_"*string(D)
    V=U₁Space(-1/2=>1,1/2=>1)
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
        p_bar = Progress(L_num,desc="M_Z",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
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
        center=fld(L[2],2)
        println("M_Z_ALL")
        println(m_data)
        println("M_Z_Center")
        println(sum(abs.(m_data[:,center+1]))/L[1])
        return [sum(abs.(m_data[1:2:end,center+1]))/L[1]*2, sum(abs.(m_data[2:2:end,center+1]))/L[1]*2]
    end
end