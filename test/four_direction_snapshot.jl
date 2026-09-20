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

engine_root=abspath(get(ENV,"ENGINE_ROOT",joinpath(@__DIR__,"..")))
include(joinpath(engine_root,"model",ENV["model"]*".jl"))
include(joinpath(engine_root,"MPS.jl"))

function main()
    haskey(ENV,"DIRECTION_SNAPSHOT_ROOT") ||
        error("DIRECTION_SNAPSHOT_ROOT is required")
    snapshot_root=abspath(ENV["DIRECTION_SNAPSHOT_ROOT"])
    passes=parse(Int,get(ENV,"DIRECTION_PASSES","4"))
    passes==4 || error("This gate requires exactly four alternating directions")

    initial_MPS_wavefunction()
    initial_step_MPS()

    D=parse(Int,ENV["BOND_DIMENSION"])
    save_file=checkpoint_path_for_dimension(D)
    source_file=save_file
    L_num=parameter["L"][1]*parameter["L"][2]

    for pass in 1:passes
        rev=iseven(pass)
        energy,truncation=MPS_sweep(source_file,save_file;D=D,ite=pass,rev=rev)
        MPI.Barrier(comm)
        if rank==root
            current=_resolve_checkpoint_directory(save_file)
            destination=joinpath(snapshot_root,
                "pass_$(pass)_$(rev ? "reverse" : "forward")")
            ispath(destination) && error("Snapshot already exists: $destination")
            cp(current,destination;force=false)
            manifest=validate_checkpoint(destination,L_num)
            get(manifest,"complete",false)==true || error("Incomplete direction snapshot")
            open(joinpath(snapshot_root,"directions.tsv"),"a") do io
                println(io,pass,'\t',rev ? "reverse" : "forward",'\t',energy,
                    '\t',truncation,'\t',manifest["metadata"]["checkpoint_sha256"])
            end
            if @isdefined ENVIRONMENT_EPOCH_MANIFEST
                epoch_path=joinpath(tmp_file,ENVIRONMENT_EPOCH_MANIFEST)
                isfile(epoch_path) && cp(epoch_path,
                    joinpath(snapshot_root,"epoch_pass_$(pass).toml");force=false)
            end
        end
        MPI.Barrier(comm)
        source_file=save_file
    end

    cleanup_environment_store!()
    rank==root && println("four_direction_snapshot=PASS passes=",passes)
    return nothing
end

main()
MPI.Finalized() || MPI.Finalize()
