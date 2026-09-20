#!/usr/bin/env julia

"""
Lossless ITensor-MPS to TensorKit-MPS conversion prototype.

The converter keeps the existing one-dimensional site sequence. ITensor suffix
charges on a bond are mapped to TensorKit prefix charges by

    q_prefix = q_total - q_suffix

for both `Nf` and `Sz` (`Sz` is the integer `Nup-Ndn`, not physical S^z).
Each output site is a TensorMap `(left * physical) ← right`, saved under the
`ttn_tem` key expected by `DMRG/MPS/TN.jl`.

This version intentionally converts one site through a dense work array. It is
appropriate for compatibility tests and a controlled D=5000 conversion on a
high-memory node, but not for future D=20000 checkpoints. A blockwise reader is
required before D=20000 conversion.
"""

using FileIO
using HDF5
using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using SHA
using TensorKit
using TOML

const U1U1 = U1Irrep ⊠ U1Irrep
const U1U1Space = Vect[U1U1]

sector_tuple(c) = Tuple(Int(s.charge) for s in c.sectors)

function qn_tuple(q::QN)
    return (val(q, "Nf"), val(q, "Sz"))
end

function physical_space_and_permutation(site::Index)
    expected = Dict((0, 0) => 1, (1, 1) => 1, (1, -1) => 1, (2, 0) => 1)
    dims, positions = transformed_space_data(site, identity)
    dims == expected || error("Unsupported local basis/QNs: $(ITensors.space(site))")
    V = U1U1Space(dims)
    return V, canonical_permutation(V, positions)
end

function transformed_space_data(index::Index, transform)
    dims = Dict{Tuple{Int,Int},Int}()
    positions = Dict{Tuple{Int,Int},Vector{Int}}()
    offset = 0
    for qdim in ITensors.space(index)
        q, multiplicity = qdim.first, qdim.second
        key = transform(qn_tuple(q))
        haskey(dims, key) && error("Charge collision while converting index $index")
        dims[key] = multiplicity
        positions[key] = collect((offset + 1):(offset + multiplicity))
        offset += multiplicity
    end
    offset == ITensors.dim(index) || error("Index dimension accounting failed for $index")
    return dims, positions
end

function canonical_permutation(space_tk, old_positions)
    permutation = Int[]
    for sector in sectors(space_tk)
        key = sector_tuple(sector)
        append!(permutation, old_positions[key])
    end
    return permutation
end

function virtual_space_and_permutation(index::Index, total_charge)
    transform(q) = (total_charge[1] - q[1], total_charge[2] - q[2])
    dims, positions = transformed_space_data(index, transform)
    space_tk = U1U1Space(dims)
    return space_tk, canonical_permutation(space_tk, positions)
end

function tagged_index(tensor::ITensor, tag::AbstractString)
    matches = filter(i -> hastags(i, tag), collect(inds(tensor)))
    length(matches) == 1 || error("Expected one '$tag' index, found $(length(matches))")
    return only(matches)
end

function site_components(psi::MPS, n::Int, total_charge)
    N = length(psi)
    A = psi[n]
    physical = tagged_index(A, "Site")
    V, physical_perm = physical_space_and_permutation(physical)

    if n == 1
        left_space = U1U1Space(Dict((0, 0) => 1))
        left_perm = [1]
    else
        left = tagged_index(A, "Link,l=$(n - 1)")
        left_space, left_perm = virtual_space_and_permutation(left, total_charge)
    end

    if n == N
        right_space = U1U1Space(Dict(total_charge => 1))
        right_perm = [1]
    else
        right = tagged_index(A, "Link,l=$n")
        right_space, right_perm = virtual_space_and_permutation(right, total_charge)
    end

    if n == 1 && n == N
        raw = reshape(Array(A, physical), 1, ITensors.dim(physical), 1)
    elseif n == 1
        right = tagged_index(A, "Link,l=$n")
        raw = reshape(Array(A, physical, right), 1, ITensors.dim(physical), ITensors.dim(right))
    elseif n == N
        left = tagged_index(A, "Link,l=$(n - 1)")
        raw = reshape(Array(A, left, physical), ITensors.dim(left), ITensors.dim(physical), 1)
    else
        left = tagged_index(A, "Link,l=$(n - 1)")
        right = tagged_index(A, "Link,l=$n")
        raw = Array(A, left, physical, right)
    end

    canonical = raw[left_perm, physical_perm, right_perm]
    return TensorMap(canonical, left_space * V, right_space),
           (left_perm, physical_perm, right_perm)
end

function source_sha256(path)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function save_atomic(tensor::TensorMap, destination::AbstractString)
    mkpath(dirname(destination))
    ispath(destination) && error("Refusing to overwrite $destination")
    temporary = destination * ".tmp.$(getpid()).jld2"
    try
        FileIO.save(temporary, "ttn_tem", tensor)
        mv(temporary, destination)
    finally
        isfile(temporary) && rm(temporary)
    end
    return destination
end

function read_itensor_mps(path::AbstractString)
    return h5open(path, "r") do file
        haskey(file, "psi") || error("No HDF5 object named 'psi' in $path")
        read(file, "psi", MPS)
    end
end

function convert_checkpoint(source::AbstractString, output_dir::AbstractString)
    isfile(source) || error("Source checkpoint does not exist: $source")
    if isdir(output_dir) && !isempty(readdir(output_dir))
        error("Refusing to write into non-empty directory: $output_dir")
    end
    mkpath(output_dir)

    psi = read_itensor_mps(source)
    total_flux = flux(psi[1])
    total_charge = qn_tuple(total_flux)
    total_charge[1] >= 0 || error("Invalid total particle number: $total_charge")

    site_records = Vector{Dict{String,Any}}()
    for n in eachindex(psi)
        tensor, _ = site_components(psi, n, total_charge)
        destination = joinpath(output_dir, "$(n).jld2")
        save_atomic(tensor, destination)
        push!(site_records, Dict(
            "site" => n,
            "left_dim" => TensorKit.dim(TensorKit.space(tensor, 1)),
            "physical_dim" => TensorKit.dim(TensorKit.space(tensor, 2)),
            "right_dim" => TensorKit.dim(TensorKit.space(tensor, 3)),
            "sectors" => length(blocksectors(tensor)),
            "file" => basename(destination),
            "bytes" => filesize(destination),
        ))
    end

    metadata = Dict{String,Any}(
        "format" => "TensorKit site TensorMap (left * physical) <- right",
        "source" => abspath(source),
        "source_bytes" => filesize(source),
        "source_sha256" => source_sha256(source),
        "length" => length(psi),
        "total_Nf" => total_charge[1],
        "total_Sz_integer" => total_charge[2],
        "itensor_llim" => psi.llim,
        "itensor_rlim" => psi.rlim,
        "local_basis" => ["Emp", "Up", "Dn", "UpDn"],
        "virtual_charge_map" => "q_prefix = q_total - q_ITensor_suffix",
        "itensors_version" => string(pkgversion(ITensors)),
        "itensormps_version" => string(pkgversion(ITensorMPS)),
        "tensorkit_version" => string(pkgversion(TensorKit)),
        "julia_version" => string(VERSION),
        "dense_workspace" => true,
        "sites" => site_records,
    )
    open(joinpath(output_dir, "conversion.toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    return metadata
end

function main(args=ARGS)
    length(args) == 2 || error("Usage: itensor_to_tensorkit.jl SOURCE.h5 OUTPUT_DIR")
    metadata = convert_checkpoint(args[1], args[2])
    length_mps = metadata["length"]
    total_nf = metadata["total_Nf"]
    total_sz = metadata["total_Sz_integer"]
    println("converted length=$length_mps ",
            "Nf=$total_nf ",
            "Sz_integer=$total_sz ",
            "output=$(abspath(args[2]))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
