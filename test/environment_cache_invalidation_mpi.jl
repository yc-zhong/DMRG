#!/usr/bin/env julia

using Test
using TensorKit
import MPI

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "..", "MPS", "TN.jl"))

configured_root = get(ENV, "ENVIRONMENT_CACHE_TEST_ROOT", "")
root = if isempty(configured_root)
    MPI.bcast(rank == 0 ? mktempdir() : nothing, 0, comm)
else
    abspath(configured_root)
end
ENV["ENV_LOCAL_CACHE_GB"] = "0.01"
ENV["ENV_VERIFY_WRITES"] = "true"
configured_owner = get(ENV, "ENVIRONMENT_STORE_OWNER", "")
ENV["ENVIRONMENT_STORE_OWNER"] = isempty(configured_owner) ?
    MPI.bcast(rank == 0 ? "cache-invalidation-test-$(getpid())" : nothing,
        0, comm) : configured_owner
configure_environment_store!(root)

MPI.Barrier(comm)
if rank == 0
    cleanup_environment_backing!()
    mkpath(root)
end
MPI.Barrier(comm)

symmetry = U1Irrep ⊠ U1Irrep
qnspace = Vect[symmetry]((0, 0) => 2, (1, 1) => 2)
reference = TensorMap(randn, Float64, qnspace ← qnspace)
reference = MPI.bcast(rank == 0 ? reference : nothing, 0, comm)
hamiltonian_key = joinpath(root, "hamiltonian_environment")
krylov_key = joinpath(root, "enviroment")

if rank == 0
    tensor_save(reference, hamiltonian_key)
    tensor_save(2.0 * reference, krylov_key)
end
MPI.Barrier(comm)

# Warm both rank-local cache entries.  Mutating a returned TensorMap must not
# alias the cached value.
hamiltonian_loaded = tensor_load(hamiltonian_key)
krylov_loaded = tensor_load(krylov_key)
@test norm(hamiltonian_loaded - reference) <= 1e-12
@test norm(krylov_loaded - 2.0 * reference) <= 1e-12
for sector in blocksectors(hamiltonian_loaded)
    blocks(hamiltonian_loaded)[sector] .= 0
end
@test norm(tensor_load(hamiltonian_key) - reference) <= 1e-12
@test environment_store_stats().cache_entries == 2
MPI.Barrier(comm)

# Reproduce the Lanczos overwrite protocol: the writer atomically publishes a
# new value, all ranks invalidate exactly that logical key, and no rank reloads
# until invalidation is globally complete.
if rank == 0
    tensor_save(3.0 * reference, krylov_key)
end
MPI.Barrier(comm)
@test environment_cache_invalidate!(krylov_key)
@test environment_store_stats().cache_entries == 1
@test haskey(_environment_cache, _environment_key(hamiltonian_key))
@test !haskey(_environment_cache, _environment_key(krylov_key))
MPI.Barrier(comm)

hits_before = environment_store_stats().cache_hits
@test norm(tensor_load(hamiltonian_key) - reference) <= 1e-12
@test environment_store_stats().cache_hits == hits_before + 1
@test norm(tensor_load(krylov_key) - 3.0 * reference) <= 1e-12

# Invalidating an absent exact key is a harmless no-op and must not evict the
# two valid cache entries.
@test !environment_cache_invalidate!(joinpath(root, "absent"))
@test environment_store_stats().cache_entries == 2

MPI.Barrier(comm)
if rank == 0
    write_zero(hamiltonian_key)
    write_zero(krylov_key)
end
MPI.Barrier(comm)
environment_cache_clear!()
MPI.Barrier(comm)
if rank == 0
    cleanup_environment_backing!(remove_roots=true)
    println("environment_cache_invalidation_mpi=PASS ranks=", nranks)
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
