using Test
using TensorKit
import MPI

include(joinpath(@__DIR__, "..", "model", "Hubbard_OBC_LeftEdge_U1_U1_core.jl"))
include(joinpath(@__DIR__, "..", "MPS", "TN.jl"))

MPI.Init()
comm=MPI.COMM_WORLD
rank=MPI.Comm_rank(comm)
nranks=MPI.Comm_size(comm)

V=HubbardOBCLeftEdgeU1U1.physical_space()
input=(rank+1)*id(V)
@test length(blocksectors(input))==4
legacy=copy(input)
packed=copy(input)
_mpi_reduce_tensormap_sum_legacy!(legacy,comm;root=0)
_mpi_reduce_tensormap_sum_packed!(packed,comm;root=0)

if rank==0
    expected=nranks*(nranks+1)/2
    @test norm(legacy-expected*id(V)) < 1e-13
    @test norm(packed-legacy) < 1e-13
    @test collect(blocksectors(packed))==collect(blocksectors(legacy))
    @test all(norm(blocks(packed)[sector]-blocks(legacy)[sector]) < 1e-13
              for sector in blocksectors(packed))
end
MPI.Barrier(comm)

# Exercise the public mode switch so validation jobs can retain an exact A/B
# path without changing engine code.
ENV["MPI_TENSORMAP_REDUCE_MODE"]="packed"
public_packed=copy(input)
mpi_reduce_tensormap_sum!(public_packed,comm;root=0)
if rank==0
    @test norm(public_packed-packed) < 1e-13
end
MPI.Barrier(comm)
