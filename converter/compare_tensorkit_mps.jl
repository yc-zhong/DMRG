#!/usr/bin/env julia

using FileIO
using LinearAlgebra
using TensorKit
using TensorOperations

"""Return the contiguous numbered site files in an engine checkpoint directory."""
function numbered_site_files(directory::AbstractString)
    isdir(directory) || error("MPS directory does not exist: $directory")
    numbered = Pair{Int,String}[]
    for name in readdir(directory)
        matched = match(r"^(\d+)\.jld2$", name)
        matched === nothing && continue
        push!(numbered, parse(Int, only(matched.captures)) => joinpath(directory, name))
    end
    isempty(numbered) && error("No numbered site files found in $directory")
    sort!(numbered; by=first)
    sites = first.(numbered)
    sites == collect(1:length(sites)) || error(
        "Site files are not the contiguous sequence 1:$(length(sites)): $sites")
    return last.(numbered)
end

load_site(path::AbstractString) = FileIO.load(path)["ttn_tem"]

"""
Contract `<bra|ket>` without converting either block-sparse MPS to a dense
many-body vector.  Both inputs use the engine convention
`(left * physical) <- right`.
"""
function mixed_inner(bra_files, ket_files)
    length(bra_files) == length(ket_files) || error("MPS lengths differ")

    bra = load_site(bra_files[1])
    ket = load_site(ket_files[1])
    dim(space(bra, 1)) == 1 || error("Bra left boundary is not one-dimensional")
    dim(space(ket, 1)) == 1 || error("Ket left boundary is not one-dimensional")
    space(bra, 2) ≅ space(ket, 2) || error("Physical spaces differ at site 1")
    @tensor environment[-1, -2] := ket[1, 2, -1] * conj(bra[1, 2, -2])

    for site in 2:length(bra_files)
        bra = load_site(bra_files[site])
        ket = load_site(ket_files[site])
        space(bra, 2) ≅ space(ket, 2) || error("Physical spaces differ at site $site")
        @tensor next_environment[-1, -2] := environment[1, 2] *
            ket[1, 3, -1] * conj(bra[2, 3, -2])
        environment = next_environment
    end

    dim(space(environment, 1)) == 1 || error("Ket right boundary is not one-dimensional")
    dim(space(environment, 2)) == 1 || error("Bra right boundary is not one-dimensional")
    return only(convert(Array, environment))
end

function checkpoint_summary(files)
    maximum_bond_dimension = 1
    maximum_blocks = 0
    for path in files
        tensor = load_site(path)
        maximum_bond_dimension = max(
            maximum_bond_dimension,
            dim(space(tensor, 1)),
            dim(space(tensor, 3)),
        )
        maximum_blocks = max(maximum_blocks, length(blocksectors(tensor)))
    end
    return maximum_bond_dimension, maximum_blocks
end

"""Require every physical and virtual TensorKit space to match exactly site by site."""
function compare_site_spaces(bra_files, ket_files)
    mismatches = String[]
    for site in eachindex(bra_files)
        bra = load_site(bra_files[site])
        ket = load_site(ket_files[site])
        for leg in 1:3
            space(bra, leg) == space(ket, leg) && continue
            push!(mismatches, "site=$(site),leg=$(leg)")
        end
    end
    return isempty(mismatches), mismatches
end

function main(args=ARGS)
    length(args) == 2 || error(
        "Usage: compare_tensorkit_mps.jl BRA_DIRECTORY KET_DIRECTORY")
    bra_directory, ket_directory = abspath.(args)
    bra_files = numbered_site_files(bra_directory)
    ket_files = numbered_site_files(ket_directory)
    length(bra_files) == length(ket_files) || error(
        "MPS lengths differ: $(length(bra_files)) and $(length(ket_files))")

    bra_norm2 = real(mixed_inner(bra_files, bra_files))
    ket_norm2 = real(mixed_inner(ket_files, ket_files))
    overlap = mixed_inner(bra_files, ket_files)
    bra_norm2 > 0 || error("Bra has non-positive norm squared: $bra_norm2")
    ket_norm2 > 0 || error("Ket has non-positive norm squared: $ket_norm2")
    normalized_overlap = abs(overlap) / sqrt(bra_norm2 * ket_norm2)
    fidelity_per_site = normalized_overlap^(1 / length(bra_files))
    bra_maxdim, bra_maxblocks = checkpoint_summary(bra_files)
    ket_maxdim, ket_maxblocks = checkpoint_summary(ket_files)
    site_spaces_match, space_mismatches = compare_site_spaces(bra_files, ket_files)

    println("bra_directory=", bra_directory)
    println("ket_directory=", ket_directory)
    println("length=", length(bra_files))
    println("bra_norm2=", bra_norm2)
    println("ket_norm2=", ket_norm2)
    println("overlap_real=", real(overlap))
    println("overlap_imag=", imag(overlap))
    println("normalized_overlap=", normalized_overlap)
    println("fidelity_per_site=", fidelity_per_site)
    println("bra_max_bond_dimension=", bra_maxdim)
    println("ket_max_bond_dimension=", ket_maxdim)
    println("bra_max_block_count=", bra_maxblocks)
    println("ket_max_block_count=", ket_maxblocks)
    println("site_spaces_match=", site_spaces_match)
    println("site_space_mismatches=", join(space_mismatches, ";"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
