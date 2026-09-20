#!/usr/bin/env julia

using MPI
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

using LinearAlgebra
using MKL
using Random
using Sockets
using TensorKit

blas_threads = parse(Int, get(ENV, "OMP_THREADS", "1"))
transformer_threads = parse(Int, get(ENV, "TENSORKIT_TRANSFORMER_THREADS", "1"))
BLAS.set_num_threads(blas_threads)
TensorKit.set_num_transformer_threads(transformer_threads)

function allowed_cpu_list()
    for line in eachline("/proc/self/status")
        startswith(line, "Cpus_allowed_list:") || continue
        return strip(split(line, ':'; limit=2)[2])
    end
    return "UNKNOWN"
end

function thread_cpu_lists()
    result = Tuple{String,String}[]
    for tid in sort(readdir("/proc/self/task"))
        status = joinpath("/proc/self/task", tid, "status")
        mask = "UNKNOWN"
        for line in eachline(status)
            startswith(line, "Cpus_allowed_list:") || continue
            mask = strip(split(line, ':'; limit=2)[2])
            break
        end
        push!(result, (tid, mask))
    end
    return result
end

host = gethostname()
affinity = allowed_cpu_list()
affinity_line = string("HYBRID_AFFINITY rank=", rank,
    " nranks=", nranks,
    " host=", host,
    " cpus_allowed=", affinity,
    " julia_threads=", Threads.nthreads(),
    " transformer_threads=", TensorKit.get_num_transformer_threads(),
    " blas_threads=", BLAS.get_num_threads())
println(affinity_line)
flush(stdout)

result_root = abspath(ENV["AFFINITY_RESULT_ROOT"])
isdir(result_root) || error("Missing affinity result root: $result_root")
result_file = joinpath(result_root, "rank_$(rank).txt")
open(result_file, "w") do io
    println(io, affinity_line)
end

# Exercise MKL with a matrix large enough to start its worker team while
# remaining a bounded smoke test.  The checksum prevents dead-code removal.
Random.seed!(0x5eed + rank)
n = parse(Int, get(ENV, "SMOKE_GEMM_N", "2048"))
A = randn(n, n)
B = randn(n, n)
C = similar(A)
mul!(C, A, B) # warm-up
MPI.Barrier(comm)
started = time_ns()
mul!(C, A, B)
elapsed = (time_ns() - started) / 1.0e9
checksum = sum(abs2, C)
gemm_line = string("HYBRID_GEMM rank=", rank,
    " n=", n,
    " seconds=", elapsed,
    " checksum=", checksum)
println(gemm_line)
flush(stdout)
open(result_file, "a") do io
    println(io, gemm_line)
    for (tid, mask) in thread_cpu_lists()
        println(io, "HYBRID_THREAD_AFFINITY rank=", rank,
            " tid=", tid,
            " cpus_allowed=", mask)
    end
end

MPI.Barrier(comm)
rank == 0 && println("hybrid_thread_affinity_smoke=PASS ranks=", nranks)
MPI.Finalize()
