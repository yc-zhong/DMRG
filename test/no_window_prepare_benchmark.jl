#!/usr/bin/env julia

import MPI
MPI.Init()
const comm=MPI.COMM_WORLD
const root=0
const rank=Int(MPI.Comm_rank(comm))
const sizeofmpi=Int(MPI.Comm_size(comm))

using LinearAlgebra
using MKL
using TensorOperations
BLAS.set_num_threads(parse(Int,ENV["OMP_THREADS"]))

include(joinpath(@__DIR__,"..","model",ENV["model"]*".jl"))
include(joinpath(@__DIR__,"..","MPS.jl"))

source=abspath(ENV["SOURCE_GENERATION"])
Len=parameter["L"][1]*parameter["L"][2]
for site in (1,Len)
    isfile(joinpath(source,string(site)*".jld2")) ||
        error("Missing source site $site in $source")
end

validate_profile_segments!(comm)
_set_environment_window_for_dimension!(parse(Int,ENV["BOND_DIMENSION"]))
environment_window_sites()==0 || error("No-window benchmark requires window=0")
validate_environment_window_mode!()
reset_segment_timings!()
environment_cache_clear!()
MPI.Barrier(comm)
if rank==root
    cleanup_environment_backing!()
    println("ENVIRONMENT_STORE_ROOT=",tmp_file)
    println("ENVIRONMENT_STORE_SPILL_ROOT=",_environment_spill_root[])
end
MPI.Barrier(comm)

started=time_ns()
mini_started=time_ns()
initial_blocks_mini(source;p=1)
mini_seconds=(time_ns()-mini_started)/1.0e9
add_segment_timing!(:initial_mini_blocks,mini_seconds)
MPI.Barrier(comm)

full_started=time_ns()
initial_blocks(source;p=1)
full_seconds=(time_ns()-full_started)/1.0e9
add_segment_timing!(:initial_full_blocks,full_seconds)
MPI.Barrier(comm)
total_seconds=(time_ns()-started)/1.0e9
add_segment_timing!(:initial_total,total_seconds)

verified=1
if rank==root
    combined=tensor_load(joinpath(tmp_file,string(hash([1,1]))))
    verified &= Int(isfinite(norm(combined)) && norm(combined)>0)
end
if rank==1%sizeofmpi
    norm_environment=tensor_load(joinpath(tmp_file,string(hash([2,1]))))
    verified &= Int(isfinite(norm(norm_environment)) && norm(norm_environment)>0)
end
verified_total=MPI.Allreduce(verified,+,comm)
verified_total==sizeofmpi || error("No-window environment verification failed")

maximum_total=MPI.Allreduce(total_seconds,max,comm)
minimum_total=MPI.Allreduce(total_seconds,min,comm)
rank==root && println("NO_WINDOW_RESULT status=PASS direction=right sites=",Len,
    " D=",ENV["BOND_DIMENSION"]," ranks=",sizeofmpi,
    " min_rank_seconds=",minimum_total," max_rank_seconds=",maximum_total,
    " combined_cut1_norm=",norm(tensor_load(joinpath(tmp_file,string(hash([1,1]))))))
report_segment_timings!("no-window-right",comm;root=root)

cleanup_environment_store!()
MPI.Finalized() || MPI.Finalize()
