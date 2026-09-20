#!/usr/bin/env julia

using FileIO
using LinearAlgebra
using TensorKit
using TensorOperations

include("../model/Hubbard_U1_U1_operators.jl")
using .HubbardU1U1Operators: hubbard_u1u1_operators

function numbered_site_files(directory::AbstractString)
    files = Pair{Int,String}[]
    for name in readdir(directory; join=false)
        matched = match(r"^(\d+)\.jld2$", name)
        matched === nothing && continue
        push!(files, parse(Int, matched.captures[1]) => joinpath(directory, name))
    end
    sort!(files; by=first)
    first.(files) == collect(1:length(files)) || error("Non-contiguous site files")
    return last.(files)
end

load_tensor(path) = convert(TensorMap, FileIO.load(path)["ttn_tem"])

function local_operator(matrix, physical_space)
    return TensorMap(Float64.(matrix), physical_space', physical_space')
end

function advance_identity(environment, tensor)
    @tensor next[-1, -2] := environment[1, 2] *
        tensor[2, 3, -2] * conj(tensor[1, 3, -1])
    return next
end

function advance_operator(environment, tensor, operator)
    @tensor next[-1, -2] := environment[1, 2] *
        tensor[2, 3, -2] * operator[3, 4] * conj(tensor[1, 4, -1])
    return next
end

function retreat_identity(environment, tensor)
    @tensor previous[-1, -2] := tensor[-2, 3, 2] *
        conj(tensor[-1, 3, 1]) * environment[1, 2]
    return previous
end

function close_environment(left, right)
    @tensor value[] := left[1, 2] * right[1, 2]
    return only(value.data)
end

function build_norm_environments(tensors)
    N = length(tensors)
    left = Vector{TensorMap}(undef, N + 1)
    right = Vector{TensorMap}(undef, N + 1)
    left[1] = id(space(tensors[1], 1))
    for site in 1:N
        left[site + 1] = advance_identity(left[site], tensors[site])
    end
    right[N + 1] = id(space(tensors[N], 3))
    for site in N:-1:1
        right[site] = retreat_identity(right[site + 1], tensors[site])
    end
    return left, right
end

function one_site_expectation(tensors, left, right, operator, site)
    inserted = advance_operator(left[site], tensors[site], operator)
    return close_environment(inserted, right[site + 1])
end

function two_site_expectation(tensors, left, right, operator_a, i, operator_b, j)
    1 <= i < j <= length(tensors) || error("Expected 1 <= i < j <= N")
    environment = advance_operator(left[i], tensors[i], operator_a)
    for site in (i + 1):(j - 1)
        environment = advance_identity(environment, tensors[site])
    end
    environment = advance_operator(environment, tensors[j], operator_b)
    return close_environment(environment, right[j + 1])
end

function parse_pairs(value, N)
    isempty(strip(value)) && return unique([(1, min(2, N)), (1, N),
                                             (max(1, N ÷ 2), min(N, N ÷ 2 + 1))])
    pairs = Tuple{Int,Int}[]
    for token in split(value, ',')
        fields = split(strip(token), ':')
        length(fields) == 2 || error("Pair must have i:j form: $token")
        i, j = parse.(Int, fields)
        i < j || error("Pair must satisfy i < j: $token")
        push!(pairs, (i, j))
    end
    return unique(pairs)
end

function main(args=ARGS)
    length(args) in (1, 2) || error(
        "Usage: measure_tensorkit_hubbard.jl MPS_DIR [i:j,k:l,...]")
    directory = abspath(args[1])
    files = numbered_site_files(directory)
    tensors = load_tensor.(files)
    N = length(tensors)
    pairs = parse_pairs(length(args) == 2 ? args[2] : "", N)
    physical = space(tensors[1], 2)
    all(space(tensor, 2) ≅ physical for tensor in tensors) ||
        error("Physical spaces are inconsistent")

    matrices = hubbard_u1u1_operators()
    n = local_operator(matrices["Ntol"], physical)
    sz = local_operator(0.5 .* matrices["Nup_minus_Ndn"], physical)
    doublon = local_operator(matrices["Nupdn"], physical)
    left, right = build_norm_environments(tensors)
    norm2 = real(close_environment(left[N + 1], right[N + 1]))

    density = real.([one_site_expectation(tensors, left, right, n, i) for i in 1:N]) ./ norm2
    spin = real.([one_site_expectation(tensors, left, right, sz, i) for i in 1:N]) ./ norm2
    double_occupancy = real.([
        one_site_expectation(tensors, left, right, doublon, i) for i in 1:N
    ]) ./ norm2
    charge_connected = Float64[]
    spin_connected = Float64[]
    for (i, j) in pairs
        nn = real(two_site_expectation(tensors, left, right, n, i, n, j)) / norm2
        ss = real(two_site_expectation(tensors, left, right, sz, i, sz, j)) / norm2
        push!(charge_connected, nn - density[i] * density[j])
        push!(spin_connected, ss - spin[i] * spin[j])
    end

    println("directory=", directory)
    println("length=", N)
    println("norm2=", norm2)
    println("total_N=", sum(density))
    println("total_Sz=", sum(spin))
    println("density=", density)
    println("spin_z=", spin)
    println("double_occupancy=", double_occupancy)
    println("pairs=", pairs)
    println("charge_connected=", charge_connected)
    println("spin_connected=", spin_connected)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
