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
    isfile(joinpath(source,string(site))) ||
        isfile(joinpath(source,string(site)*".bin")) ||
        isfile(joinpath(source,string(site)*".jld2")) ||
        error("Missing source site $site in $source")
end

validate_profile_segments!(comm)
_set_environment_window_for_dimension!(parse(Int,ENV["BOND_DIMENSION"]))
validate_environment_window_mode!()
if rank==root
    println("ENVIRONMENT_STORE_ROOT=",tmp_file)
    println("ENVIRONMENT_STORE_SPILL_ROOT=",_environment_spill_root[])
end
reset_segment_timings!()
_reset_window_work_counters!(:right)

MPI.Barrier(comm)
started=time_ns()
prepare_environment_window!(false,source,1;initialize_anchors=true)
local_seconds=(time_ns()-started)/1.0e9
maximum_seconds=MPI.Allreduce(local_seconds,max,comm)
minimum_seconds=MPI.Allreduce(local_seconds,min,comm)
rank==root && println("FIRST_WINDOW_RESULT status=PASS direction=forward bonds=1:32",
    " sites=",Len," D=",ENV["BOND_DIMENSION"],
    " ranks=",sizeofmpi," min_rank_seconds=",minimum_seconds,
    " max_rank_seconds=",maximum_seconds,
    " environment_site_updates=",_window_environment_site_updates[],
    " anchor_snapshots=",_window_anchor_snapshots[])
report_segment_timings!("first-window-forward",comm;root=root)

# This benchmark intentionally stops before H-action and never writes an MPS.
# Anchors are benchmark artifacts inside the owned environment root.
cleanup_environment_store!()
MPI.Finalized() || MPI.Finalize()
