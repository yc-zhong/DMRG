using Test
using SHA
using TensorKit

include(joinpath(@__DIR__, "..", "model", "Hubbard_OBC_LeftEdge_U1_U1_core.jl"))
using .HubbardOBCLeftEdgeU1U1
include(joinpath(@__DIR__, "..", "MPS", "TN.jl"))

@testset "read-only import and atomic checkpoint generations" begin
    adapter=build_adapter(2,2;Ne=4,Sz2=0)
    tensors=initial_mps_tensors(adapter)
    parent=mktempdir()
    source=joinpath(parent,"source")
    target=joinpath(parent,"target")
    mkpath(source)
    for (site,tensor) in enumerate(tensors)
        tensor_save(tensor,joinpath(source,string(site)))
    end
    source_hashes=[open(SHA.sha256,joinpath(source,"$(site).jld2")) for site in 1:4]
    source_identity=checkpoint_identity(source,4)

    incoming=begin_checkpoint_generation(source,target,4;generation="import")
    commit_checkpoint_generation(target,incoming,4)
    @test source_hashes==[open(SHA.sha256,joinpath(source,"$(site).jld2")) for site in 1:4]
    @test validate_checkpoint(target,4;
        physical_space=adapter.V[1],left_boundary=adapter.V_in[1],right_boundary=adapter.V_out[1]) !== nothing
    @test checkpoint_identity(target,4)==source_identity

    current_before=strip(read(joinpath(target,CHECKPOINT_CURRENT),String))
    stable_hash=checkpoint_identity(target,4)
    interrupted=begin_checkpoint_generation(target,target,4;generation="interrupted")
    open(joinpath(interrupted,"1.jld2"),"w") do io
        write(io,"intentionally incomplete generation")
    end
    @test strip(read(joinpath(target,CHECKPOINT_CURRENT),String))==current_before
    @test checkpoint_identity(target,4)==stable_hash

    replacement=begin_checkpoint_generation(target,target,4;generation="replacement")
    write_checkpoint_manifest(replacement,4;generation="replacement",
        metadata=Dict("source_sha256"=>source_identity))
    commit_checkpoint_generation(target,replacement,4)
    @test isfile(joinpath(target,CHECKPOINT_PREVIOUS))
    @test validate_checkpoint(target,4) !== nothing

    current_dir=_resolve_checkpoint_directory(target)
    open(joinpath(current_dir,"1.jld2"),"w") do io
        write(io,"broken current generation")
    end
    @test_throws Exception validate_checkpoint(target,4)
    failed=recover_checkpoint_previous(target,4)
    @test failed !== nothing && isdir(failed)
    @test validate_checkpoint(target,4) !== nothing
    @test source_hashes==[open(SHA.sha256,joinpath(source,"$(site).jld2")) for site in 1:4]

    bad=joinpath(parent,"bad")
    mkpath(bad)
    for site in 1:4
        tensor_save(site==1 ? tensors[2] : tensors[site],joinpath(bad,string(site)))
    end
    @test_throws ErrorException validate_checkpoint(bad,4;
        physical_space=adapter.V[1],left_boundary=adapter.V_in[1],right_boundary=adapter.V_out[1])
end

