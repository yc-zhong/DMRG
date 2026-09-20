using Test
using LinearAlgebra
using ITensors
using ITensorMPS

# Run from /Users/kani/Workplace. This uses the separate reader/reference
# environment because the production engine environment intentionally does not
# depend on ITensor:
#   julia +1.11.5 --project=DMRG/converter DMRG/test/hubbard_obc_itensor_ed.jl

include("hubbard_obc_ed_reference.jl")
using .HubbardOBCEDReference

function add_reference_hopping!(os, i, j, amplitude)
    os += -amplitude, "Cdagup", i, "Cup", j
    os += -amplitude, "Cdagup", j, "Cup", i
    os += -amplitude, "Cdagdn", i, "Cdn", j
    os += -amplitude, "Cdagdn", j, "Cdn", i
    return os
end

function reference_itensor_mpo(
    sites;
    Lx,
    Ly,
    t=1.0,
    ty=1.0,
    U=0.0,
    left_edge_h=0.0,
)
    os = OpSum()
    for bond in nearest_neighbor_bonds_reference(Lx, Ly; t=t, ty=ty)
        os = add_reference_hopping!(os, bond.i, bond.j, bond.amplitude)
    end
    for x in 1:Lx, y in 1:Ly
        i = site_index_reference(x, y, Lx, Ly)
        os += U, "Nupdn", i
        field = left_edge_h * left_edge_profile_reference(x, y, Lx, Ly)
        if !iszero(field)
            # The production convention is h_i*(Nup-Ndn), not h_i*Sz.
            os += field, "Nup", i
            os += -field, "Ndn", i
        end
    end
    return MPO(os, sites)
end

function state_labels(up::UInt, dn::UInt, nsites::Integer)
    labels = Vector{String}(undef, nsites)
    for site in 1:nsites
        has_up = !iszero(up & (UInt(1) << (site - 1)))
        has_dn = !iszero(dn & (UInt(1) << (site - 1)))
        labels[site] = has_up ? (has_dn ? "UpDn" : "Up") : (has_dn ? "Dn" : "Emp")
    end
    return labels
end

function fixed_sector_itensor_matrix(H::MPO, sites, basis)
    states = [MPS(sites, state_labels(up, dn, length(sites))) for (up, dn) in basis]
    matrix = zeros(ComplexF64, length(basis), length(basis))
    for column in eachindex(states), row in eachindex(states)
        matrix[row, column] = inner(states[row]', H, states[column])
    end
    return matrix
end

"""
Phase for changing from the reference basis
`(all up orbitals) tensor (all down orbitals)` to ITensor's site-major Electron
basis `(up_1,down_1,up_2,down_2,...)`. It only fixes basis convention; it is
not fitted from either Hamiltonian.
"""
function species_major_to_site_major_phase(up::UInt, dn::UInt, nsites::Integer)
    inversions = 0
    for i in 1:nsites
        has_dn = !iszero(dn & (UInt(1) << (i - 1)))
        has_dn || continue
        for j in (i + 1):nsites
            inversions += !iszero(up & (UInt(1) << (j - 1)))
        end
    end
    return isodd(inversions) ? -1.0 : 1.0
end

@testset "2x2 U=12 OBC left-edge Hubbard: independent ED vs ITensor" begin
    Lx, Ly = 2, 2
    Nup, Ndn = 2, 2
    t, ty = 1.0, 0.7
    U = 12.0
    left_edge_h = 0.1

    H_reference, basis = dense_fixed_sector_hamiltonian(
        Lx, Ly, Nup, Ndn;
        t=t,
        ty=ty,
        U=U,
        left_edge_h=left_edge_h,
    )
    sites = siteinds("Electron", Lx * Ly; conserve_qns=true)
    H_mpo = reference_itensor_mpo(
        sites;
        Lx=Lx,
        Ly=Ly,
        t=t,
        ty=ty,
        U=U,
        left_edge_h=left_edge_h,
    )
    H_itensor = fixed_sector_itensor_matrix(H_mpo, sites, basis)

    phases = [species_major_to_site_major_phase(up, dn, Lx * Ly) for (up, dn) in basis]
    basis_change = Diagonal(phases)
    matrix_error = maximum(abs.(H_itensor - basis_change * H_reference * basis_change))

    reference_spectrum = eigvals(Hermitian(H_reference))
    itensor_spectrum = eigvals(Hermitian(H_itensor))
    matrix_hermiticity_error = norm(H_itensor - H_itensor')
    spectrum_error = maximum(abs.(reference_spectrum - itensor_spectrum))

    @test matrix_hermiticity_error < 1e-12
    @test matrix_error < 1e-12
    @test spectrum_error < 1e-11
    @test reference_spectrum[1:8] ≈ itensor_spectrum[1:8] atol=1e-11 rtol=1e-11

    println("fixed_sector_dimension=", length(basis))
    println("itensor_matrix_hermiticity_error=", matrix_hermiticity_error)
    println("basis_aligned_matrix_max_error=", matrix_error)
    println("full_spectrum_max_error=", spectrum_error)
    println("lowest_eigenvalues=", reference_spectrum[1:8])
end
