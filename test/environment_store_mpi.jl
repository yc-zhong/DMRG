#!/usr/bin/env julia

import MPI
using TensorKit

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "..", "MPS", "TN.jl"))

root = abspath(haskey(ENV, "ENVIRONMENT_STORE_TEST_ROOT") ?
    ENV["ENVIRONMENT_STORE_TEST_ROOT"] : mktempdir())
ENV["ENV_LOCAL_CACHE_GB"] = get(ENV, "ENV_LOCAL_CACHE_GB", "0.01")
ENV["ENV_VERIFY_WRITES"] = get(ENV, "ENV_VERIFY_WRITES", "true")
configure_environment_store!(root)
force_spill = lowercase(get(ENV, "ENV_STORE_TEST_FORCE_SPILL", "false")) in
    ("1", "true", "yes", "on")

MPI.Barrier(comm)
if rank == 0
    cleanup_environment_backing!()
    mkpath(root)
end
MPI.Barrier(comm)

symmetry = U1Irrep ⊠ U1Irrep
qnspace = Vect[symmetry]((0, 0) => 2, (1, 1) => 3, (1, -1) => 2)
reference = TensorMap(randn, Float64, qnspace ← qnspace)
reference = MPI.bcast(rank == 0 ? reference : nothing, 0, comm)
key = joinpath(root, "cross_rank_tensor")

if rank == 0
    tensor_save(reference, key)
    if force_spill
        spill_key = joinpath(abspath(ENV["ENVIRONMENT_SPILL_ROOT"]), "cross_rank_tensor")
        @assert !isfile(key * ".bin")
        @assert isfile(spill_key * ".bin")
    end
end
MPI.Barrier(comm)

loaded = tensor_load(key)
@assert space(loaded) == space(reference)
@assert norm(loaded - reference) <= 1e-12
loaded_again = tensor_load(key)
@assert norm(loaded_again - reference) <= 1e-12

clone_source = joinpath(root, "clone_source")
clone_key = joinpath(root, "clone_snapshot")
moved_key = joinpath(root, "clone_restored")
if rank == 0
    tensor_save(3.0 * reference, clone_source)
    cloned = environment_clone_backing!(clone_source, clone_key)
    @assert cloned.mode in (:hardlink, :copy)
    @assert cloned.bytes > 0
    # Atomic replacement of the active key must not mutate the immutable
    # snapshot, including when it is a hard link to the previous inode.
    tensor_save(4.0 * reference, clone_source)
end
MPI.Barrier(comm)
cloned_value = tensor_load(clone_key)
@assert norm(cloned_value - 3.0 * reference) <= 1e-12
MPI.Barrier(comm)
if rank == 0
    moved = environment_move_backing!(clone_key, moved_key)
    @assert moved.bytes > 0
end
MPI.Barrier(comm)
moved_value = tensor_load(moved_key)
@assert norm(moved_value - 3.0 * reference) <= 1e-12
MPI.Barrier(comm)
if rank == 0
    @assert !isfile(clone_key * ".bin")
    @assert !isfile(clone_key * ".jld2")
    write_zero(clone_source)
    write_zero(moved_key)
end
MPI.Barrier(comm)

ham_key = joinpath(root, "eigenvectors")
if rank == 0
    Ham_save([reference, 2.0 * reference], ham_key)
end
MPI.Barrier(comm)
loaded_vector = tensor_load(ham_key)
@assert loaded_vector isa Vector
@assert length(loaded_vector) == 2
@assert norm(loaded_vector[1] - reference) <= 1e-12
@assert norm(loaded_vector[2] - 2.0 * reference) <= 1e-12

if rank == 0
    tensor_save(2.0 * reference, key)
end
MPI.Barrier(comm)
environment_cache_clear!()
MPI.Barrier(comm)

updated = tensor_load(key)
@assert norm(updated - 2.0 * reference) <= 1e-12
MPI.Barrier(comm)

# Candidate tiered ping-pong primitives: cold write, exact promote, exact
# demote, and cross-rank readback. Relocation is only legal at this barriered
# no-reader point.
if !isempty(environment_store_stats().spill_root)
    tier_key=joinpath(root,"tiered_tensor")
    if rank==0
        with_environment_write_preference(:cold) do
            tensor_save(5.0*reference,tier_key)
        end
        spill_key=joinpath(environment_store_stats().spill_root,"tiered_tensor")
        @assert !isfile(tier_key*".bin")
        @assert isfile(spill_key*".bin")
        promoted=environment_relocate_backing!(tier_key,:primary)
        @assert promoted.mode in (:promoted,:capacity_skip)
        if promoted.mode==:promoted
            @assert isfile(tier_key*".bin")
            @assert !isfile(spill_key*".bin")
        else
            @assert !isfile(tier_key*".bin")
            @assert isfile(spill_key*".bin")
        end
    end
    MPI.Barrier(comm)
    promoted_value=tensor_load(tier_key)
    @assert space(promoted_value)==space(reference)
    @assert norm(promoted_value-5.0*reference)<=1e-12
    MPI.Barrier(comm)
    environment_cache_clear!()
    if rank==0
        demoted=environment_relocate_backing!(tier_key,:spill)
        @assert demoted.mode in (:demoted,:already_present)
    end
    MPI.Barrier(comm)
    demoted_value=tensor_load(tier_key)
    @assert space(demoted_value)==space(reference)
    @assert norm(demoted_value-5.0*reference)<=1e-12
    MPI.Barrier(comm)
    rank==0 && write_zero(tier_key)
    MPI.Barrier(comm)
end

if rank == 0
    write_zero(key)
    write_zero(key)
    @assert !isfile(key * ".bin")
    @assert !isfile(key * ".jld2")
    @assert !isfile(dirname(key) * ".jld2")
    if force_spill
        spill_key = joinpath(abspath(ENV["ENVIRONMENT_SPILL_ROOT"]), "cross_rank_tensor")
        @assert !isfile(spill_key * ".bin")
        @assert !isfile(spill_key * ".jld2")
    end
end
MPI.Barrier(comm)

stats = environment_store_stats()
@assert stats.cache_hits >= 1
println("ENV_STORE_TEST rank=", rank, " stats=", stats)

MPI.Barrier(comm)
if rank == 0
    cleanup_environment_backing!(remove_roots=true)
    println("environment_store_mpi=PASS ranks=", nranks)
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
