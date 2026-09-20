using Test
using LinearAlgebra
using TensorKit

# Run from /Users/kani/Workplace:
#   julia +1.11.5 --project=DMRG DMRG/test/hubbard_obc_adapter_ed.jl

include("hubbard_obc_ed_reference.jl")
include("../model/Hubbard_OBC_LeftEdge_U1_U1_core.jl")

const RefED = HubbardOBCEDReference
const Adapter = HubbardOBCLeftEdgeU1U1

canonical_reference_bonds(Lx, Ly; t, ty) = [
    (i=b.i, j=b.j, axis=b.axis, coupling=b.amplitude)
    for b in RefED.nearest_neighbor_bonds_reference(Lx, Ly; t=t, ty=ty)
]

@testset "OBC column-snake geometry and left_edge_af convention" begin
    Lx, Ly = 3, 2
    for x in 1:Lx, y in 1:Ly
        site = RefED.site_index_reference(x, y, Lx, Ly)
        @test Adapter.site_index(x, y, Lx, Ly) == site
        @test Adapter.xy_from_site(site, Lx, Ly) == (x, y)
        @test Adapter.pinning_profile(x, y, Lx, Ly) ==
              RefED.left_edge_profile_reference(x, y, Lx, Ly)
    end

    t, ty = 0.9, 0.7
    adapter_bonds = sort(Adapter.nearest_neighbor_bonds(Lx, Ly; t=t, ty=ty);
                         by=b -> (b.i, b.j, b.axis))
    reference_bonds = canonical_reference_bonds(Lx, Ly; t=t, ty=ty)
    @test adapter_bonds == reference_bonds
    @test length(adapter_bonds) == (Lx - 1) * Ly + Lx * (Ly - 1)
    @test length(unique((b.i, b.j) for b in adapter_bonds)) == length(adapter_bonds)
end

@testset "exact pruned and fused hopping channels" begin
    split = Adapter.build_adapter(
        3, 2; U=12.0, Ne=6, Sz2=0, hopping_channel_mode=:split_pruned,
    )
    fused = Adapter.build_adapter(
        3, 2; U=12.0, Ne=6, Sz2=0, hopping_channel_mode=:fused_pruned,
    )

    @test split.hopping_channel_mode == :split_pruned
    @test fused.hopping_channel_mode == :fused_pruned
    @test length(split.hopping_factors) == 2
    @test length(fused.hopping_factors) == 1
    @test length(split.terms) == 2length(split.bonds)
    @test length(fused.terms) == length(fused.bonds)
    @test dim(domain(split.hopping_factors[1][1])) == 2
    @test dim(domain(split.hopping_factors[2][1])) == 2
    @test dim(domain(fused.hopping_factors[1][1])) == 4

    split_sum = sum(permute(pair[1] * pair[2], (1, 3), (2, 4))
        for pair in split.hopping_factors)
    fused_map = permute(
        fused.hopping_factors[1][1] * fused.hopping_factors[1][2],
        (1, 3), (2, 4),
    )
    exact_sum = split.hopping_maps[1] + split.hopping_maps[2]
    @test norm(split_sum - exact_sum) < 1e-12
    @test norm(fused_map - exact_sum) < 1e-12
    @test norm(fused_map - split_sum) < 1e-12
end

@testset "32x6 production sectors and exact open-bond inventory" begin
    Lx, Ly = 32, 6
    bonds = Adapter.nearest_neighbor_bonds(Lx, Ly; t=1.0, ty=1.0)
    @test length(bonds) == 346
    @test count(b -> b.axis == :x, bonds) == 186
    @test count(b -> b.axis == :y, bonds) == 160
    @test length(unique((b.i, b.j) for b in bonds)) == 346
    @test all(b -> 1 <= b.i < b.j <= Lx * Ly, bonds)

    @test [Adapter.pinning_profile(1, y, Lx, Ly) for y in 1:Ly] ==
          [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    @test all(iszero(Adapter.pinning_profile(x, y, Lx, Ly))
              for x in 2:Lx for y in 1:Ly)

    @test Adapter.target_particle_numbers(Lx * Ly, 192, 0) == (96, 96)
    @test Adapter.target_particle_numbers(Lx * Ly, 180, 0) == (90, 90)
    for Ne in (192, 180)
        charges = Adapter.initial_local_charges(Lx, Ly, Ne, 0)
        @test length(charges) == 192
        @test sum(first, charges) == Ne
        @test sum(last, charges) == 0
    end
end

@testset "U=0 fixed-N exact single-particle benchmark" begin
    Lx, Ly = 3, 2
    Nup, Ndn = 3, 3
    t, ty = 0.9, 0.7
    left_edge_h = 0.1

    for spin in (-1, 1)
        h_adapter = Adapter.single_particle_matrix(
            Lx, Ly; t=t, ty=ty, spin=spin, left_edge_h=left_edge_h,
        )
        h_reference = RefED.single_particle_matrix_reference(
            Lx, Ly; t=t, ty=ty, spin=spin, left_edge_h=left_edge_h,
        )
        @test h_adapter == h_reference
        @test ishermitian(h_adapter)
    end

    energy_adapter = Adapter.exact_u0_energy(
        Lx, Ly, Nup, Ndn; t=t, ty=ty, left_edge_h=left_edge_h,
    )
    energy_reference = RefED.exact_u0_energy_reference(
        Lx, Ly, Nup, Ndn; t=t, ty=ty, left_edge_h=left_edge_h,
    )
    H_manybody, _ = RefED.dense_fixed_sector_hamiltonian(
        Lx, Ly, Nup, Ndn; t=t, ty=ty, U=0.0, left_edge_h=left_edge_h,
    )
    energy_manybody = eigmin(Hermitian(H_manybody))

    @test energy_adapter ≈ energy_reference atol=1e-13 rtol=1e-13
    @test energy_adapter ≈ energy_manybody atol=1e-12 rtol=1e-12
    println("u0_fixed_sector_dimension=", size(H_manybody, 1))
    println("u0_exact_energy=", energy_adapter)
    println("u0_manybody_energy_error=", abs(energy_adapter - energy_manybody))
end

@testset "adapter term tables and onsite matrices" begin
    Lx, Ly = 3, 2
    t, ty = 0.9, 0.7
    U = 12.0
    left_edge_h = 0.1
    data = Adapter.build_adapter(
        Lx, Ly;
        t=t,
        ty=ty,
        U=U,
        left_edge_h=left_edge_h,
        Ne=Lx * Ly,
        Sz2=0,
    )

    @test sort(data.bonds; by=b -> (b.i, b.j, b.axis)) ==
          canonical_reference_bonds(Lx, Ly; t=t, ty=ty)
    @test length(data.terms) == 2length(data.bonds)
    @test maximum(data.factorization_errors) < 1e-12
    @test sort(collect(keys(data.Ham_matrix))) == [-2, -1]

    for bond in data.bonds, channel in 1:2
        key = [bond.i, bond.j, -channel]
        @test haskey(data.terms, key)
        @test data.terms[key] == (bond.axis == :x ? "t" : "ty")
    end

    nupdn = Diagonal([0.0, 0.0, 0.0, 1.0])
    nup_minus_ndn = Diagonal([0.0, 1.0, -1.0, 0.0])
    for x in 1:Lx, y in 1:Ly
        site = Adapter.site_index(x, y, Lx, Ly)
        profile = RefED.left_edge_profile_reference(x, y, Lx, Ly)
        expected = Matrix(U * nupdn + left_edge_h * profile * nup_minus_ndn)
        @test convert(Array, data.terms_onsite[site]) == expected
    end

    for channel in 1:2
        left, right = data.hopping_factors[channel]
        @test dim(domain(left)) == 2
        @test dim(codomain(right)) == 2
        reconstructed = permute(left * right, (1, 3), (2, 4))
        @test norm(reconstructed - data.hopping_maps[channel]) < 1e-12
    end
end
