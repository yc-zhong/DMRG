module HubbardOBCEDReference

using LinearAlgebra

export site_index_reference,
       xy_from_site_reference,
       nearest_neighbor_bonds_reference,
       left_edge_profile_reference,
       single_particle_matrix_reference,
       exact_u0_energy_reference,
       fixed_sector_basis,
       dense_fixed_sector_hamiltonian

"""Independent column-snake map for an `Lx x Ly` OBC rectangle."""
function site_index_reference(x::Integer, y::Integer, Lx::Integer, Ly::Integer)
    1 <= x <= Lx || throw(BoundsError(1:Lx, x))
    1 <= y <= Ly || throw(BoundsError(1:Ly, y))
    return isodd(x) ? (x - 1) * Ly + y : x * Ly - y + 1
end

function xy_from_site_reference(i::Integer, Lx::Integer, Ly::Integer)
    1 <= i <= Lx * Ly || throw(BoundsError(1:(Lx * Ly), i))
    x = div(i - 1, Ly) + 1
    offset = i - (x - 1) * Ly
    y = isodd(x) ? offset : Ly - offset + 1
    return (x, y)
end

"""
Return physical nearest-neighbor bonds as `(i, j, amplitude, axis)` records.
Every bond is emitted once, with `i < j`; no periodic wrap bonds are present.
"""
function nearest_neighbor_bonds_reference(Lx::Integer, Ly::Integer; t=1.0, ty=1.0)
    bonds = NamedTuple{(:i, :j, :amplitude, :axis),Tuple{Int,Int,Float64,Symbol}}[]
    for x in 1:(Lx - 1), y in 1:Ly
        i = site_index_reference(x, y, Lx, Ly)
        j = site_index_reference(x + 1, y, Lx, Ly)
        push!(bonds, (i=min(i, j), j=max(i, j), amplitude=Float64(t), axis=:x))
    end
    for x in 1:Lx, y in 1:(Ly - 1)
        i = site_index_reference(x, y, Lx, Ly)
        j = site_index_reference(x, y + 1, Lx, Ly)
        push!(bonds, (i=min(i, j), j=max(i, j), amplitude=Float64(ty), axis=:y))
    end
    sort!(bonds; by=b -> (b.i, b.j, b.axis))
    return bonds
end

"""
Coefficient multiplying `Nup-Ndn` for the production single-left-edge AF field.
The returned profile is `(-1)^(x+y)` on `x=1`, and zero elsewhere.
It is deliberately not a coefficient of `Sz=(Nup-Ndn)/2`.
"""
function left_edge_profile_reference(x::Integer, y::Integer, Lx::Integer, Ly::Integer)
    1 <= x <= Lx || throw(BoundsError(1:Lx, x))
    1 <= y <= Ly || throw(BoundsError(1:Ly, y))
    return x == 1 ? (iseven(x + y) ? 1.0 : -1.0) : 0.0
end

function single_particle_matrix_reference(
    Lx::Integer,
    Ly::Integer;
    t=1.0,
    ty=1.0,
    spin::Integer,
    left_edge_h=0.0,
)
    spin in (-1, 1) || throw(ArgumentError("spin must be +1 (up) or -1 (down)"))
    h = zeros(Float64, Lx * Ly, Lx * Ly)
    for bond in nearest_neighbor_bonds_reference(Lx, Ly; t=t, ty=ty)
        h[bond.i, bond.j] -= bond.amplitude
        h[bond.j, bond.i] -= bond.amplitude
    end
    for x in 1:Lx, y in 1:Ly
        i = site_index_reference(x, y, Lx, Ly)
        h[i, i] += spin * left_edge_h * left_edge_profile_reference(x, y, Lx, Ly)
    end
    return h
end

function exact_u0_energy_reference(
    Lx::Integer,
    Ly::Integer,
    Nup::Integer,
    Ndn::Integer;
    t=1.0,
    ty=1.0,
    left_edge_h=0.0,
)
    nsites = Lx * Ly
    0 <= Nup <= nsites || throw(ArgumentError("invalid Nup=$Nup"))
    0 <= Ndn <= nsites || throw(ArgumentError("invalid Ndn=$Ndn"))
    eup = eigvals(Hermitian(single_particle_matrix_reference(
        Lx, Ly; t=t, ty=ty, spin=1, left_edge_h=left_edge_h,
    )))
    edn = eigvals(Hermitian(single_particle_matrix_reference(
        Lx, Ly; t=t, ty=ty, spin=-1, left_edge_h=left_edge_h,
    )))
    return sum(eup[1:Nup]) + sum(edn[1:Ndn])
end

"""Lexicographically ordered `(up_bits, down_bits)` fixed-sector basis."""
function fixed_sector_basis(nsites::Integer, Nup::Integer, Ndn::Integer)
    0 <= Nup <= nsites || throw(ArgumentError("invalid Nup=$Nup"))
    0 <= Ndn <= nsites || throw(ArgumentError("invalid Ndn=$Ndn"))
    limit = UInt(1) << nsites
    up = [bits for bits in UInt(0):(limit - UInt(1)) if count_ones(bits) == Nup]
    dn = [bits for bits in UInt(0):(limit - UInt(1)) if count_ones(bits) == Ndn]
    return [(u, d) for u in up for d in dn]
end

@inline occupied(bits::UInt, site::Integer) = !iszero(bits & (UInt(1) << (site - 1)))

"""Return `(new_bits, sign)` for `c^dagger_i c_j`, or `nothing` if forbidden."""
function apply_hop(bits::UInt, i::Integer, j::Integer)
    occupied(bits, j) || return nothing
    occupied(bits, i) && return nothing
    lo, hi = minmax(i, j)
    between_mask = hi - lo <= 1 ? UInt(0) : xor(
        (UInt(1) << (hi - 1)) - UInt(1),
        (UInt(1) << lo) - UInt(1),
    )
    sign = isodd(count_ones(bits & between_mask)) ? -1.0 : 1.0
    result = xor(xor(bits, UInt(1) << (j - 1)), UInt(1) << (i - 1))
    return result, sign
end

"""
Dense physical Hubbard Hamiltonian in a fixed `(Nup,Ndn)` sector.

The basis factors the two conserved spin species. This is sufficient because
the tested Hamiltonian contains no pairing or spin-flip term. The onsite field
is exactly `h_i*(Nup-Ndn)`, matching the production ITensor convention.
"""
function dense_fixed_sector_hamiltonian(
    Lx::Integer,
    Ly::Integer,
    Nup::Integer,
    Ndn::Integer;
    t=1.0,
    ty=1.0,
    U=0.0,
    left_edge_h=0.0,
)
    nsites = Lx * Ly
    basis = fixed_sector_basis(nsites, Nup, Ndn)
    index = Dict(state => k for (k, state) in pairs(basis))
    H = zeros(Float64, length(basis), length(basis))

    for (column, (up, dn)) in pairs(basis)
        double_occupancy = count_ones(up & dn)
        diagonal = Float64(U) * double_occupancy
        for x in 1:Lx, y in 1:Ly
            site = site_index_reference(x, y, Lx, Ly)
            profile = left_edge_profile_reference(x, y, Lx, Ly)
            diagonal += left_edge_h * profile *
                        (Int(occupied(up, site)) - Int(occupied(dn, site)))
        end
        H[column, column] += diagonal

        for bond in nearest_neighbor_bonds_reference(Lx, Ly; t=t, ty=ty)
            for (source, destination) in ((bond.i, bond.j), (bond.j, bond.i))
                up_hop = apply_hop(up, destination, source)
                if !isnothing(up_hop)
                    new_up, sign = up_hop
                    row = index[(new_up, dn)]
                    H[row, column] += -bond.amplitude * sign
                end
                dn_hop = apply_hop(dn, destination, source)
                if !isnothing(dn_hop)
                    new_dn, sign = dn_hop
                    row = index[(up, new_dn)]
                    H[row, column] += -bond.amplitude * sign
                end
            end
        end
    end

    return H, basis
end

end
