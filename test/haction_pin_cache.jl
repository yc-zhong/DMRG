#!/usr/bin/env julia

using Test
using TensorKit

include(joinpath(@__DIR__, "..", "MPS", "TN.jl"))

ENV["ENV_LOCAL_CACHE_GB"]="0"
ENV["ENV_VERIFY_WRITES"]="true"
ENV["ENVIRONMENT_STORE_OWNER"]="haction-pin-cache-test-$(getpid())"
temporary_parent=mktempdir()
# Production rejects broad one-level /tmp roots.  Keep the test subject to the
# same safety rule by placing its owned environment below the random parent.
root=joinpath(temporary_parent,"owned_environment")
configure_environment_store!(root)

symmetry=U1Irrep ⊠ U1Irrep
space_qn=Vect[symmetry]((0,0)=>3,(1,1)=>2)
environment=TensorMap(randn,Float64,space_qn←space_qn)
probe=TensorMap(randn,Float64,space_qn←space_qn)
hamiltonian_key=joinpath(root,"hamiltonian_environment")
krylov_key=joinpath(root,"enviroment")
tensor_save(environment,hamiltonian_key)
tensor_save(2.0*environment,krylov_key)

function block_fingerprint(tensor)
    return Dict(sector=>copy(blocks(tensor)[sector]) for sector in blocksectors(tensor))
end

function fingerprint_equal(tensor,fingerprint)
    collect(blocksectors(tensor))==collect(keys(fingerprint)) ||
        Set(blocksectors(tensor))==Set(keys(fingerprint)) || return false
    return all(blocks(tensor)[sector]==fingerprint[sector] for sector in keys(fingerprint))
end

baseline=environment*probe
first_identity=Ref{Any}(nothing)
source_fingerprint=block_fingerprint(environment)
result=with_haction_pin_cache(7;forbidden_keys=(krylov_key,),
        budget_bytes=1_000_000,emit=false) do
    pinned=haction_environment_load_readonly(hamiltonian_key)
    first_identity[]=pinned
    pinned_fingerprint=block_fingerprint(pinned)

    # Repeated H-actions reuse exactly the pinned object and remain numerically
    # identical to the uncached contraction.
    outputs=TensorMap[]
    for _ in 1:3
        source=haction_environment_load_readonly(hamiltonian_key)
        @test source===pinned
        push!(outputs,source*probe)
        @test fingerprint_equal(source,pinned_fingerprint)
    end
    @test all(norm(output-baseline)<=1e-13 for output in outputs)

    # The production before/after path multiplies by a scalar before pre_add!
    # mutates the accumulator. Verify that operation allocates independent
    # block storage and leaves the pinned source bitwise unchanged.
    accumulator=haction_environment_load_readonly(hamiltonian_key)*2.0
    for sector in blocksectors(accumulator)
        blocks(accumulator)[sector].=0
    end
    @test fingerprint_equal(pinned,pinned_fingerprint)

    # A mutable Krylov scratch key is forbidden: it is never admitted or
    # aliased even though it lives below the environment root.
    mutable_one=haction_environment_load_readonly(krylov_key)
    mutable_two=haction_environment_load_readonly(krylov_key)
    @test mutable_one !== mutable_two
    @test norm(mutable_one-2.0*environment)<=1e-13
    @test norm(mutable_two-2.0*environment)<=1e-13
    return outputs[end]
end
@test norm(result-baseline)<=1e-13
stats=_haction_pin_last_stats[]
@test stats.status==:complete
@test stats.admissions==1
@test stats.misses==1
@test stats.hits==4
@test stats.rejections==0
@test stats.peak_bytes<=stats.budget_bytes
@test !_haction_pin_active[]
@test isempty(_haction_pin_cache)
@test fingerprint_equal(environment,source_fingerprint)

# Strict over-budget fallback performs normal independent loads and admits
# nothing; no resident bytes can exceed the configured limit.
fallback_identity=with_haction_pin_cache(8;budget_bytes=1,emit=false) do
    a=haction_environment_load_readonly(hamiltonian_key)
    b=haction_environment_load_readonly(hamiltonian_key)
    @test a !== b
    @test norm(a*probe-baseline)<=1e-13
    @test norm(b*probe-baseline)<=1e-13
    return a===b
end
@test !fallback_identity
fallback_stats=_haction_pin_last_stats[]
@test fallback_stats.admissions==0
@test fallback_stats.rejections==2
@test fallback_stats.bytes==0
@test fallback_stats.peak_bytes==0

# Scope cleanup is exception-safe and therefore cannot leak a pin into the
# next bond.
@test_throws ErrorException with_haction_pin_cache(9;
        budget_bytes=1_000_000,emit=false) do
    haction_environment_load_readonly(hamiltonian_key)
    error("intentional pin-cache scope failure")
end
@test _haction_pin_last_stats[].status==:exception
@test !_haction_pin_active[]
@test isempty(_haction_pin_cache)
@test _haction_pin_bytes[]==0

write_zero(hamiltonian_key)
write_zero(krylov_key)
cleanup_environment_backing!(remove_roots=true)
println("haction_pin_cache=PASS")
