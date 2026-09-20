module HubbardOBCLeftEdgeU1U1

using LinearAlgebra
using TensorKit
using TensorOperations

include("Hubbard_U1_U1_operators.jl")
using .HubbardU1U1Operators: hubbard_u1u1_operators

export AdapterData,
       build_adapter,
       exact_u0_energy,
       initial_local_charges,
       nearest_neighbor_bonds,
       pinning_profile,
       single_particle_matrix,
       site_index,
       target_particle_numbers,
       tensor_operator,
       xy_from_site

const U1U1Sector = Irrep[U₁] ⊠ Irrep[U₁]
const U1U1Space = Vect[U1U1Sector]

physical_space() = U1U1Space(
    (0, 0) => 1,
    (2, 0) => 1,
    (1, 1) => 1,
    (1, -1) => 1,
)

charge_space(charge::Tuple{Int,Int}) = U1U1Space(charge => 1)

struct AdapterData
    Lx::Int
    Ly::Int
    t::Float64
    ty::Float64
    U::Float64
    left_edge_h::Float64
    Ne::Int
    Sz2::Int
    Vsite
    V
    V_in
    V_out
    hopping_maps::Vector{TensorMap}
    hopping_factors::Vector{Vector{TensorMap}}
    factorization_errors::Vector{Float64}
    hopping_channel_mode::Symbol
    Ham_matrix::Dict{Int,Vector{TensorMap}}
    terms::Dict{Vector{Int},String}
    terms_onsite::Dict{Int,TensorMap}
    bonds::Vector{NamedTuple{(:i, :j, :axis, :coupling),Tuple{Int,Int,Symbol,Float64}}}
end

function validate_geometry(Lx::Int, Ly::Int)
    Lx > 0 || throw(ArgumentError("Lx must be positive, got $Lx"))
    Ly > 0 || throw(ArgumentError("Ly must be positive, got $Ly"))
    return nothing
end

"""Column-snake site index used by the production ITensor workflow."""
function site_index(x::Int, y::Int, Lx::Int, Ly::Int)
    validate_geometry(Lx, Ly)
    1 <= x <= Lx || throw(BoundsError(1:Lx, x))
    1 <= y <= Ly || throw(BoundsError(1:Ly, y))
    return isodd(x) ? (x - 1) * Ly + y : (x - 1) * Ly + (Ly - y + 1)
end

function xy_from_site(i::Int, Lx::Int, Ly::Int)
    validate_geometry(Lx, Ly)
    1 <= i <= Lx * Ly || throw(BoundsError(1:(Lx * Ly), i))
    x = div(i - 1, Ly) + 1
    offset = i - (x - 1) * Ly
    y = isodd(x) ? offset : Ly - offset + 1
    return x, y
end

"""Return every nearest-neighbor bond exactly once for OBC in x and y."""
function nearest_neighbor_bonds(Lx::Int, Ly::Int; t::Real=1.0, ty::Real=1.0)
    validate_geometry(Lx, Ly)
    bond_type = NamedTuple{(:i, :j, :axis, :coupling),Tuple{Int,Int,Symbol,Float64}}
    bonds = bond_type[]
    for y in 1:Ly, x in 1:(Lx - 1)
        a = site_index(x, y, Lx, Ly)
        b = site_index(x + 1, y, Lx, Ly)
        push!(bonds, (i=min(a, b), j=max(a, b), axis=:x, coupling=Float64(t)))
    end
    for x in 1:Lx, y in 1:(Ly - 1)
        a = site_index(x, y, Lx, Ly)
        b = site_index(x, y + 1, Lx, Ly)
        push!(bonds, (i=min(a, b), j=max(a, b), axis=:y, coupling=Float64(ty)))
    end
    expected = (Lx - 1) * Ly + Lx * (Ly - 1)
    length(bonds) == expected || error("Internal bond-count mismatch")
    length(unique((b.i, b.j) for b in bonds)) == expected || error("Duplicate nearest-neighbor bond")
    return bonds
end

"""left_edge_af profile multiplying Nup-Ndn, not physical Sz."""
function pinning_profile(x::Int, y::Int, Lx::Int, Ly::Int)
    site_index(x, y, Lx, Ly)
    x == 1 || return 0.0
    return isodd(x + y) ? -1.0 : 1.0
end

function target_particle_numbers(nsites::Int, Ne::Int, Sz2::Int)
    0 <= Ne <= 2nsites || throw(ArgumentError("Ne=$Ne is outside 0:$(2nsites)"))
    iseven(Ne + Sz2) || throw(ArgumentError("Ne+Sz2 must be even"))
    iseven(Ne - Sz2) || throw(ArgumentError("Ne-Sz2 must be even"))
    Nup = div(Ne + Sz2, 2)
    Ndn = div(Ne - Sz2, 2)
    0 <= Nup <= nsites || throw(ArgumentError("Nup=$Nup is outside 0:$nsites"))
    0 <= Ndn <= nsites || throw(ArgumentError("Ndn=$Ndn is outside 0:$nsites"))
    return Nup, Ndn
end

function tensor_operator(name::String; value::Real=1.0, Vsite=physical_space())
    operators = hubbard_u1u1_operators()
    haskey(operators, name) || throw(ArgumentError("Unknown local operator $name"))
    return TensorMap(Float64(value) * operators[name], Vsite', Vsite')
end

function hopping_maps(Vsite=physical_space())
    operators = hubbard_u1u1_operators()
    @tensor hopping_up[-1, -2, -3, -4] :=
        operators["C_dagup"][-1, 1] * operators["F"][1, -3] * operators["C_up"][-2, -4] -
        operators["C_up"][-1, 1] * operators["F"][1, -3] * operators["C_dagup"][-2, -4]
    @tensor hopping_dn[-1, -2, -3, -4] :=
        operators["C_dagdn"][-1, -3] * operators["F"][-2, 1] * operators["C_dn"][1, -4] -
        operators["C_dn"][-1, -3] * operators["F"][-2, 1] * operators["C_dagdn"][1, -4]
    return TensorMap[
        TensorMap(-hopping_up, Vsite' * Vsite', Vsite' * Vsite'),
        TensorMap(-hopping_dn, Vsite' * Vsite', Vsite' * Vsite'),
    ]
end

function exact_hopping_factorization(hopping::TensorMap)
    # The local hopping operator has exact algebraic rank two for each spin.
    # TensorKit's untruncated SVD nevertheless retains roundoff-scale singular
    # values (and therefore a 16-dimensional auxiliary space).  Removing only
    # values below a norm-scaled floating-point threshold is representation
    # cleanup, not a physics cutoff.  The reconstruction check below remains
    # the authority: fail rather than silently accept a lossy factorization.
    threshold = 64 * eps(Float64) * max(norm(hopping), 1.0)
    left, singular, right, discarded =
        tsvd(hopping, (1, 3), (2, 4); trunc=truncbelow(threshold))
    factors = TensorMap[left, singular * right]
    reconstructed = permute(factors[1] * factors[2], (1, 3), (2, 4))
    error = norm(reconstructed - hopping)
    tolerance = 1e-12 * max(norm(hopping), 1.0)
    error <= tolerance || throw(ArgumentError(
        "Exact hopping factorization failed: error=$error tolerance=$tolerance " *
        "threshold=$threshold discarded=$discarded",
    ))
    return factors, error
end

function fuse_hopping_factorizations(factors::Vector{Vector{TensorMap}})
    length(factors) == 2 || throw(ArgumentError(
        "Expected separate up/down hopping factors, got $(length(factors)) channels",
    ))
    left = catdomain(factors[1][1], factors[2][1])
    right = catcodomain(factors[1][2], factors[2][2])
    return TensorMap[left, right]
end

function build_terms(bonds, nchannels::Int)
    nchannels > 0 || throw(ArgumentError("nchannels must be positive"))
    terms = Dict{Vector{Int},String}()
    for bond in bonds
        coefficient = bond.axis == :x ? "t" : "ty"
        for channel in 1:nchannels
            terms[[bond.i, bond.j, -channel]] = coefficient
        end
    end
    return terms
end

function build_onsite_terms(Lx::Int, Ly::Int, U::Real, left_edge_h::Real, Vsite)
    nupdn = tensor_operator("Nupdn"; Vsite=Vsite)
    magnetization = tensor_operator("Nup_minus_Ndn"; Vsite=Vsite)
    onsite = Dict{Int,TensorMap}()
    for x in 1:Lx, y in 1:Ly
        i = site_index(x, y, Lx, Ly)
        field = Float64(left_edge_h) * pinning_profile(x, y, Lx, Ly)
        onsite[i] = Float64(U) * nupdn + field * magnetization
    end
    return onsite
end

function build_adapter(
    Lx::Int,
    Ly::Int;
    t::Real=1.0,
    ty::Real=1.0,
    U::Real=12.0,
    left_edge_h::Real=0.1,
    Ne::Int=Lx * Ly,
    Sz2::Int=0,
    hopping_channel_mode::Symbol=:split_pruned,
)
    validate_geometry(Lx, Ly)
    target_particle_numbers(Lx * Ly, Ne, Sz2)
    all(isfinite, (t, ty, U, left_edge_h)) || throw(ArgumentError("Hamiltonian coefficients must be finite"))

    Vsite = physical_space()
    V = ProductSpace(Vsite)
    V_in = ProductSpace(charge_space((0, 0)))
    V_out = ProductSpace(charge_space((Ne, Sz2)))
    maps = hopping_maps(Vsite)
    factor_results = exact_hopping_factorization.(maps)
    split_factors = first.(factor_results)
    errors = last.(factor_results)
    hopping_channel_mode in (:split_pruned, :fused_pruned) || throw(ArgumentError(
        "hopping_channel_mode must be :split_pruned or :fused_pruned",
    ))
    factors = hopping_channel_mode == :split_pruned ? split_factors :
        [fuse_hopping_factorizations(split_factors)]
    if hopping_channel_mode == :fused_pruned
        reconstructed = permute(factors[1][1] * factors[1][2], (1, 3), (2, 4))
        fused_error = norm(reconstructed - (maps[1] + maps[2]))
        fused_error <= 1e-12 * max(norm(maps[1] + maps[2]), 1.0) ||
            throw(ArgumentError("Exact fused hopping factorization failed: $fused_error"))
    end
    Ham_matrix = Dict(-channel => factors[channel] for channel in eachindex(factors))
    bonds = nearest_neighbor_bonds(Lx, Ly; t=t, ty=ty)
    terms = build_terms(bonds, length(factors))
    onsite = build_onsite_terms(Lx, Ly, U, left_edge_h, Vsite)

    return AdapterData(
        Lx,
        Ly,
        Float64(t),
        Float64(ty),
        Float64(U),
        Float64(left_edge_h),
        Ne,
        Sz2,
        Vsite,
        V,
        V_in,
        V_out,
        maps,
        factors,
        errors,
        hopping_channel_mode,
        Ham_matrix,
        terms,
        onsite,
        bonds,
    )
end

function single_particle_matrix(
    Lx::Int,
    Ly::Int;
    t::Real=1.0,
    ty::Real=1.0,
    spin::Int=1,
    left_edge_h::Real=0.0,
)
    spin in (-1, 1) || throw(ArgumentError("spin must be +1 or -1"))
    matrix = zeros(Float64, Lx * Ly, Lx * Ly)
    for bond in nearest_neighbor_bonds(Lx, Ly; t=t, ty=ty)
        matrix[bond.i, bond.j] -= bond.coupling
        matrix[bond.j, bond.i] -= bond.coupling
    end
    for x in 1:Lx, y in 1:Ly
        i = site_index(x, y, Lx, Ly)
        matrix[i, i] += spin * Float64(left_edge_h) * pinning_profile(x, y, Lx, Ly)
    end
    return matrix
end

function exact_u0_energy(
    Lx::Int,
    Ly::Int,
    Nup::Int,
    Ndn::Int;
    t::Real=1.0,
    ty::Real=1.0,
    left_edge_h::Real=0.0,
)
    nsites = Lx * Ly
    0 <= Nup <= nsites || throw(ArgumentError("Invalid Nup=$Nup"))
    0 <= Ndn <= nsites || throw(ArgumentError("Invalid Ndn=$Ndn"))
    up = eigvals(Hermitian(single_particle_matrix(
        Lx, Ly; t=t, ty=ty, spin=1, left_edge_h=left_edge_h,
    )))
    down = eigvals(Hermitian(single_particle_matrix(
        Lx, Ly; t=t, ty=ty, spin=-1, left_edge_h=left_edge_h,
    )))
    return sum(sort(up)[1:Nup]) + sum(sort(down)[1:Ndn])
end

function uniform_positions(N::Int, count::Int)
    count <= 0 && return Int[]
    count >= N && return collect(1:N)
    positions = unique(clamp.(round.(Int, range(1, N, length=count)), 1, N))
    candidate = 1
    while length(positions) < count
        candidate in positions || push!(positions, candidate)
        candidate += 1
    end
    return sort(positions)
end

function initial_local_charges(Lx::Int, Ly::Int, Ne::Int, Sz2::Int)
    nsites = Lx * Ly
    Nup, Ndn = target_particle_numbers(nsites, Ne, Sz2)
    states = fill((0, 0), nsites)
    if Ne <= nsites
        occupied = uniform_positions(nsites, Ne)
        nup_left, ndn_left = Nup, Ndn
        for i in occupied
            x, y = xy_from_site(i, Lx, Ly)
            prefer_up = iseven(x + y)
            if (prefer_up && nup_left > 0) || ndn_left == 0
                states[i] = (1, 1)
                nup_left -= 1
            else
                states[i] = (1, -1)
                ndn_left -= 1
            end
        end
    else
        states .= Ref((2, 0))
        for i in uniform_positions(nsites, nsites - Nup)
            states[i] = (1, -1)
        end
        for i in reverse(uniform_positions(nsites, nsites - Ndn))
            states[i] = states[i] == (2, 0) ? (1, 1) : (0, 0)
        end
    end
    sum(first, states) == Ne || error("Initial-state particle count mismatch")
    sum(last, states) == Sz2 || error("Initial-state spin count mismatch")
    return states
end

function initial_mps_tensors(data::AdapterData)
    local_charges = initial_local_charges(data.Lx, data.Ly, data.Ne, data.Sz2)
    tensors = Vector{TensorMap}(undef, length(local_charges))
    previous = data.V_in
    physical = data.V
    prefix_N = 0
    prefix_Sz2 = 0
    for i in eachindex(local_charges)
        prefix_N += local_charges[i][1]
        prefix_Sz2 += local_charges[i][2]
        after = ProductSpace(charge_space((prefix_N, prefix_Sz2)))
        tensors[i] = isometry(Float64, previous * physical, after)
        previous = after
    end
    return tensors
end

export initial_mps_tensors

end
