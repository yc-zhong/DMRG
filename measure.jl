using LinearAlgebra
using TensorOperations
include("MPS/TN.jl")
include("measure/Slt_CORR.jl")
BLAS.set_num_threads(4)

for D in [8000]
    mc=CORR_slt(D,"Square_Hubbard_U1_SU2_CBC_Rand/MPS[4.0, 48.0, 166.0, 8.0, 0.0]",[4,48])
end

# for D in [10000]
#     norm_TN2(1,"TNc/Square_Hubbard_Z2_U1_CBC_P/MPS[4.0, 48.0, 1.75, 8.0, 0.0, -0.25]_$D",L=[4,48])
#     mc=Delta(D,"Square_Hubbard_Z2_U1_CBC_P/MPS[4.0, 48.0, 1.75, 8.0, 0.0, -0.25]",[4,48])
# end