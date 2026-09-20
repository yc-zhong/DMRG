#!/usr/bin/env julia

import FileIO
import JLD2
import MPI
using LinearAlgebra
using MKL
using TensorKit
using TensorOperations

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

function check_dense_contraction()
    a = reshape(collect(1.0:12.0), 3, 4)
    b = reshape(collect(1.0:20.0), 4, 5)
    @tensor c[i, k] := a[i, j] * b[j, k]
    @assert c == a * b
    return norm(c)
end

function check_u1u1_blocks()
    @assert isdefined(TensorKit, :tsvd)
    @assert isdefined(TensorKit, :MatrixAlgebra)
    @assert isdefined(TensorKit.MatrixAlgebra, :svd!)
    @assert isdefined(TensorKit, :leftorth)
    @assert isdefined(TensorKit, :rightorth)

    symmetry = U1Irrep ⊠ U1Irrep
    space = Vect[symmetry]((0, 0) => 2, (1, 1) => 3, (1, -1) => 2)
    tensor = TensorMap(randn, Float64, space ← space)
    sectors_before = collect(blocksectors(tensor))
    @assert length(sectors_before) == 3

    product = tensor * adjoint(tensor)
    dense_error = norm(convert(Array, product) -
                       convert(Array, tensor) * adjoint(convert(Array, tensor)))
    @assert dense_error <= 1e-12

    left, singular, right = tsvd(tensor, (1,), (2,))
    reconstruction_error = norm(left * singular * right - tensor) / max(norm(tensor), eps())
    @assert reconstruction_error <= 1e-12

    temporary = tempname() * ".jld2"
    try
        FileIO.save(temporary, "tensor", tensor)
        restored = FileIO.load(temporary)["tensor"]
        @assert norm(restored - tensor) <= 1e-12
    finally
        isfile(temporary) && rm(temporary)
    end

    return (; sectors=length(sectors_before), dense_error, reconstruction_error)
end

dense_norm = check_dense_contraction()
block_result = check_u1u1_blocks()

rank_value = Float64(rank + 1)
rank_sum = MPI.Allreduce(rank_value, +, comm)
expected_sum = nranks * (nranks + 1) / 2
@assert rank_sum == expected_sum
MPI.Barrier(comm)

if rank == 0
    println("julia_version=", VERSION)
    println("tensorkit_version=", pkgversion(TensorKit))
    println("tensoroperations_version=", pkgversion(TensorOperations))
    println("mpi_version=", pkgversion(MPI))
    println("mkl_version=", pkgversion(MKL))
    println("blas_config=", BLAS.get_config())
    println("blas_threads=", BLAS.get_num_threads())
    println("mpi_library=", MPI.identify_implementation())
    println("mpi_ranks=", nranks)
    println("mpi_rank_sum=", rank_sum)
    println("dense_contraction_norm=", dense_norm)
    println("u1u1_sector_count=", block_result.sectors)
    println("u1u1_dense_error=", block_result.dense_error)
    println("u1u1_svd_reconstruction_error=", block_result.reconstruction_error)
    println("environment_smoke=PASS")
end

MPI.Finalized() || MPI.Finalize()
