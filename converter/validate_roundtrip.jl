#!/usr/bin/env julia

using FileIO
using HDF5
using ITensors
using ITensorMPS
using LinearAlgebra
using TensorKit
using TensorOperations

include("itensor_to_tensorkit.jl")

function inverse_site_tensor(source::MPS, n::Int, converted::TensorMap, total_charge)
    _, permutations = site_components(source, n, total_charge)
    left_perm, physical_perm, right_perm = permutations
    canonical = convert(Array, converted)
    raw = canonical[invperm(left_perm), invperm(physical_perm), invperm(right_perm)]

    A = source[n]
    physical = tagged_index(A, "Site")
    N = length(source)
    if n == 1 && n == N
        return ITensor(reshape(raw, ITensors.dim(physical)), physical)
    elseif n == 1
        right = tagged_index(A, "Link,l=$n")
        return ITensor(reshape(raw, ITensors.dim(physical), ITensors.dim(right)), physical, right)
    elseif n == N
        left = tagged_index(A, "Link,l=$(n - 1)")
        return ITensor(reshape(raw, ITensors.dim(left), ITensors.dim(physical)), left, physical)
    else
        left = tagged_index(A, "Link,l=$(n - 1)")
        right = tagged_index(A, "Link,l=$n")
        return ITensor(raw, left, physical, right)
    end
end

function selected_correlators(psi::MPS, opname::String, pairs)
    selected_sites = sort(unique(collect(Iterators.flatten(pairs))))
    positions = Dict(site => n for (n, site) in enumerate(selected_sites))
    matrix = correlation_matrix(psi, opname, opname; sites=selected_sites)
    values = Dict{String,ComplexF64}()
    for (i, j) in pairs
        values["$i,$j"] = matrix[positions[i], positions[j]]
    end
    return values
end

function max_dict_difference(a, b)
    keys(a) == keys(b) || error("Mismatched correlator keys")
    return maximum(abs(a[k] - b[k]) for k in keys(a); init=0.0)
end

function tensorkit_norm2(converted_dir, N)
    A = FileIO.load(joinpath(converted_dir, "1.jld2"))["ttn_tem"]
    @tensor environment[-1, -2] := A[1, 2, -1] * conj(A[1, 2, -2])
    for n in 2:N
        A = FileIO.load(joinpath(converted_dir, "$(n).jld2"))["ttn_tem"]
        @tensor next_environment[-1, -2] := environment[1, 2] *
            A[1, 3, -1] * conj(A[2, 3, -2])
        environment = next_environment
    end
    return only(convert(Array, environment))
end

function site_index(x, y, Lx, Ly)
    return isodd(x) ? (x - 1) * Ly + y : (x - 1) * Ly + (Ly - y + 1)
end

function target_hubbard_mpo(sites; Lx, Ly, U, pinning_strength)
    length(sites) == Lx * Ly || error("Lx*Ly does not match the MPS length")
    os = OpSum()
    function add_hop!(i, j)
        os += -1.0, "Cdagup", i, "Cup", j
        os += -1.0, "Cdagup", j, "Cup", i
        os += -1.0, "Cdagdn", i, "Cdn", j
        os += -1.0, "Cdagdn", j, "Cdn", i
    end
    for y in 1:Ly, x in 1:(Lx - 1)
        add_hop!(site_index(x, y, Lx, Ly), site_index(x + 1, y, Lx, Ly))
    end
    for y in 1:(Ly - 1), x in 1:Lx
        add_hop!(site_index(x, y, Lx, Ly), site_index(x, y + 1, Lx, Ly))
    end
    for i in eachindex(sites)
        os += U, "Nupdn", i
    end
    for y in 1:Ly
        stagger = isodd(1 + y) ? -1.0 : 1.0
        i = site_index(1, y, Lx, Ly)
        os += pinning_strength * stagger, "Nup", i
        os += -pinning_strength * stagger, "Ndn", i
    end
    return MPO(os, sites)
end

function validate_roundtrip(source_path, converted_dir; hamiltonian=nothing)
    manifest_path = joinpath(converted_dir, "conversion.toml")
    isfile(manifest_path) || error("Missing conversion manifest: $manifest_path")
    manifest = TOML.parsefile(manifest_path)
    actual_sha256 = source_sha256(source_path)
    manifest["source_sha256"] == actual_sha256 || error("Source checksum does not match conversion manifest")

    source = read_itensor_mps(source_path)
    total_charge = qn_tuple(flux(source[1]))
    manifest["length"] == length(source) || error("MPS length does not match conversion manifest")
    manifest["total_Nf"] == total_charge[1] || error("Nf does not match conversion manifest")
    manifest["total_Sz_integer"] == total_charge[2] || error("Sz does not match conversion manifest")
    restored_tensors = ITensor[]
    max_tensor_difference = 0.0

    for n in eachindex(source)
        path = joinpath(converted_dir, "$(n).jld2")
        converted = FileIO.load(path)["ttn_tem"]
        restored = inverse_site_tensor(source, n, converted, total_charge)
        push!(restored_tensors, restored)
        max_tensor_difference = max(max_tensor_difference, norm(restored - source[n]))
    end

    restored = MPS(restored_tensors)
    norm_source = real(ITensorMPS.inner(source, source))
    norm_restored = real(ITensorMPS.inner(restored, restored))
    norm_tensorkit = tensorkit_norm2(converted_dir, length(source))
    overlap = ITensorMPS.inner(source, restored)
    normalized_overlap = abs(overlap) / sqrt(norm_source * norm_restored)

    density_source = expect(source, "Ntot")
    density_restored = expect(restored, "Ntot")
    sz_source = expect(source, "Sz")
    sz_restored = expect(restored, "Sz")
    doublon_source = expect(source, "Nupdn")
    doublon_restored = expect(restored, "Nupdn")

    N = length(source)
    pairs = unique([(1, min(2, N)), (1, N), (max(1, N ÷ 2), min(N, N ÷ 2 + 1))])
    charge_source = selected_correlators(source, "Ntot", pairs)
    charge_restored = selected_correlators(restored, "Ntot", pairs)
    spin_source = selected_correlators(source, "Sz", pairs)
    spin_restored = selected_correlators(restored, "Sz", pairs)

    println("norm_source=", norm_source)
    println("norm_restored=", norm_restored)
    println("norm_tensorkit=", norm_tensorkit)
    println("normalized_overlap=", normalized_overlap)
    println("max_tensor_difference=", max_tensor_difference)
    println("total_N_source=", sum(density_source))
    println("total_N_restored=", sum(density_restored))
    println("total_Sz_source=", sum(sz_source))
    println("total_Sz_restored=", sum(sz_restored))
    println("max_local_density_difference=", maximum(abs.(density_source .- density_restored)))
    println("max_local_Sz_difference=", maximum(abs.(sz_source .- sz_restored)))
    println("max_double_occupancy_difference=", maximum(abs.(doublon_source .- doublon_restored)))
    println("max_charge_correlator_difference=", max_dict_difference(charge_source, charge_restored))
    println("max_spin_correlator_difference=", max_dict_difference(spin_source, spin_restored))

    tolerance = 5e-11
    if hamiltonian !== nothing
        energy_source = real(ITensorMPS.inner(source', hamiltonian, source))
        energy_restored = real(ITensorMPS.inner(restored', hamiltonian, restored))
        println("energy_source=", energy_source)
        println("energy_restored=", energy_restored)
        println("energy_difference=", abs(energy_source - energy_restored))
        abs(energy_source - energy_restored) <= tolerance || error("Energy validation failed")
    end

    normalized_overlap >= 1 - tolerance || error("Round-trip overlap failed")
    abs(norm_tensorkit - norm_source) <= tolerance || error("TensorKit-native norm failed")
    max_tensor_difference <= tolerance || error("Tensor round-trip failed")
    maximum(abs.(density_source .- density_restored)) <= tolerance || error("Density validation failed")
    maximum(abs.(sz_source .- sz_restored)) <= tolerance || error("Sz validation failed")
    maximum(abs.(doublon_source .- doublon_restored)) <= tolerance || error("Double occupancy validation failed")
    max_dict_difference(charge_source, charge_restored) <= tolerance || error("Charge correlator validation failed")
    max_dict_difference(spin_source, spin_restored) <= tolerance || error("Spin correlator validation failed")
end

length(ARGS) in (2, 6) || error(
    "Usage: validate_roundtrip.jl SOURCE.h5 CONVERTED_DIR [Lx Ly U left_edge_h]")
source = read_itensor_mps(ARGS[1])
hamiltonian = if length(ARGS) == 6
    target_hubbard_mpo(siteinds(source);
        Lx=parse(Int, ARGS[3]), Ly=parse(Int, ARGS[4]),
        U=parse(Float64, ARGS[5]), pinning_strength=parse(Float64, ARGS[6]))
else
    nothing
end
validate_roundtrip(ARGS[1], ARGS[2]; hamiltonian)
