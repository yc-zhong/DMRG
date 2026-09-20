import MPI
MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank=Int(MPI.Comm_rank(comm))
const sizeofmpi=Int(MPI.Comm_size(comm))

using LinearAlgebra
using MKL
using TensorKit
using TensorOperations
const _blas_threads=parse(Int,get(ENV,"OMP_THREADS","1"))
const _transformer_threads=parse(Int,get(ENV,"TENSORKIT_TRANSFORMER_THREADS","1"))
_blas_threads>=1 || error("OMP_THREADS must be positive")
_transformer_threads>=1 || error("TENSORKIT_TRANSFORMER_THREADS must be positive")
BLAS.set_num_threads(_blas_threads)
TensorKit.set_num_transformer_threads(_transformer_threads)
rank==root && println("THREAD_CONFIGURATION julia=",Threads.nthreads(),
    " tensor_transformer=",TensorKit.get_num_transformer_threads(),
    " blas=",BLAS.get_num_threads())

if isfile("model/"*ENV["model"]*".jl")
    include("model/"*ENV["model"]*".jl")
else
    throw("model file not find, please add model file to model directory")
end

include("MPS.jl")

#transfer_type_wavefunciton()

#SWEEP=vcat([100 for i in 1:10],[1000 for i in 1:10],[2000 for i in 1:10])
#SWEEP=[10000]

initial_MPS_wavefunction()

if _parse_bool("FIRST_PASS_REVERSE",get(ENV,"FIRST_PASS_REVERSE","false"))
    error("FIRST_PASS_REVERSE=true requires left-environment initialization and is not supported")
end

initial_step_MPS()

if _parse_bool("ENERGY_ONLY",get(ENV,"ENERGY_ONLY","false"))
    current_mps_energy(position=parse(Int,get(ENV,"ENERGY_ONLY_POSITION","1")))
    cleanup_environment_store!()
elseif @isdefined SWEEP
    MPS_SP()
else
    MPS()
end
