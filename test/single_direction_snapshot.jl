#!/usr/bin/env julia

import MPI
MPI.Init()
const comm=MPI.COMM_WORLD
const root=0
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

engine_root=abspath(ENV["ENGINE_ROOT"])
include(joinpath(engine_root,"model",ENV["model"]*".jl"))
include(joinpath(engine_root,"MPS.jl"))

function main()
    initial_MPS_wavefunction()
    initial_step_MPS()

    D=parse(Int,ENV["BOND_DIMENSION"])
    save_file=checkpoint_path_for_dimension(D)
    energy,truncation=MPS_sweep(save_file,save_file;D=D,ite=1,rev=false)
    MPI.Barrier(comm)
    if rank==root
        current=_resolve_checkpoint_directory(save_file)
        manifest=validate_checkpoint(current,parameter["L"][1]*parameter["L"][2])
        get(manifest,"complete",false)==true || error("Incomplete forward checkpoint")
        println("SINGLE_DIRECTION_RESULT direction=forward energy=",energy,
            " truncation=",truncation,
            " checkpoint_sha256=",manifest["metadata"]["checkpoint_sha256"])
    end
    MPI.Barrier(comm)
    cleanup_environment_store!()
    rank==root && println("single_direction_snapshot=PASS direction=forward")
    return nothing
end

main()
MPI.Finalized() || MPI.Finalize()
