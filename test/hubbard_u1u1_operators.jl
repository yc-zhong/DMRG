using Test
using LinearAlgebra
using TensorKit
using TensorOperations

include("../model/Hubbard_U1_U1_operators.jl")
using .HubbardU1U1Operators

function hopping_parts(operators)
    @tensor hopping_up[-1, -2, -3, -4] :=
        operators["C_dagup"][-1, 1] * operators["F"][1, -3] * operators["C_up"][-2, -4] -
        operators["C_up"][-1, 1] * operators["F"][1, -3] * operators["C_dagup"][-2, -4]
    @tensor hopping_dn[-1, -2, -3, -4] :=
        operators["C_dagdn"][-1, -3] * operators["F"][-2, 1] * operators["C_dn"][1, -4] -
        operators["C_dn"][-1, -3] * operators["F"][-2, 1] * operators["C_dagdn"][1, -4]
    return hopping_up, hopping_dn
end

@testset "canonical U1xU1 Hubbard operators" begin
    @test HUBBARD_U1U1_LEGACY_BASIS == ("Emp", "UpDn", "Up", "Dn")
    @test HUBBARD_U1U1_CANONICAL_BASIS == ("Emp", "Up", "Dn", "UpDn")

    legacy = hubbard_u1u1_legacy_operators()
    operators = hubbard_u1u1_operators()
    p = collect(HUBBARD_U1U1_CANONICAL_FROM_LEGACY)

    for name in keys(legacy)
        @test operators[name] == legacy[name][p, p]
    end

    @test operators["Ntol"] == Diagonal([0.0, 1.0, 1.0, 2.0])
    @test operators["Nupdn"] == Diagonal([0.0, 0.0, 0.0, 1.0])
    @test operators["Nup"] == Diagonal([0.0, 1.0, 0.0, 1.0])
    @test operators["Ndn"] == Diagonal([0.0, 0.0, 1.0, 1.0])
    @test operators["Nup_minus_Ndn"] == Diagonal([0.0, 1.0, -1.0, 0.0])
    @test operators["Sz"] == Diagonal([0.0, 0.5, -0.5, 0.0])
    @test operators["F"] == Diagonal([1.0, -1.0, -1.0, 1.0])
    @test operators["C_dagup"] == operators["C_up"]'
    @test operators["C_dagdn"] == operators["C_dn"]'

    identity4 = Matrix{Float64}(I, 4, 4)
    for spin in ("up", "dn")
        c = operators["C_$(spin)"]
        cdag = operators["C_dag$(spin)"]
        @test c * c == zeros(4, 4)
        @test c * cdag + cdag * c == identity4
        @test operators["F"] * c == -c * operators["F"]
    end

    @test operators["C_up"] * operators["C_dn"] ==
          operators["C_dn"] * operators["C_up"]
    @test operators["C_up"] * operators["C_dagdn"] ==
          operators["C_dagdn"] * operators["C_up"]

    nup = operators["C_dagup"] * operators["C_up"]
    ndn = operators["C_dagdn"] * operators["C_dn"]
    @test operators["Ntol"] == nup + ndn
    @test operators["Nup"] == nup
    @test operators["Ndn"] == ndn
    @test operators["Nup_minus_Ndn"] == nup - ndn
    @test operators["Nupdn"] == nup * ndn
    @test operators["Sz"] == (nup - ndn) / 2

    physical_cdn = (identity4 - 2nup) * operators["C_dn"]
    @test operators["C_up"] * physical_cdn + physical_cdn * operators["C_up"] == zeros(4, 4)
    @test operators["C_up"] * physical_cdn' + physical_cdn' * operators["C_up"] == zeros(4, 4)

    hopping_up_old, hopping_dn_old = hopping_parts(legacy)
    hopping_up, hopping_dn = hopping_parts(operators)
    @test hopping_up == hopping_up_old[p, p, p, p]
    @test hopping_dn == hopping_dn_old[p, p, p, p]

    hopping_matrix = reshape(-(hopping_up + hopping_dn), 16, 16)
    @test ishermitian(hopping_matrix)
    @test eigvals(Hermitian(hopping_matrix)) ≈ [
        -2.0,
        -1.0, -1.0, -1.0, -1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0,
    ] atol = 1e-14

    V = Vect[(Irrep[U₁] ⊠ Irrep[U₁])]((0, 0) => 1, (2, 0) => 1, (1, 1) => 1, (1, -1) => 1)
    sector_charges = [(sector.sectors[1].charge, sector.sectors[2].charge) for sector in sectors(V)]
    @test sector_charges == [(0, 0), (1, 1), (1, -1), (2, 0)]

    for name in ("F", "Sz", "Ntol", "Nupdn")
        @test TensorMap(operators[name], V', V') isa TensorMap
    end
    hopping_up_map = TensorMap(-hopping_up, V' * V', V' * V')
    hopping_dn_map = TensorMap(-hopping_dn, V' * V', V' * V')
    @test hopping_up_map isa TensorMap
    @test hopping_dn_map isa TensorMap

    for hopping in (hopping_up_map, hopping_dn_map)
        threshold = 64 * eps(Float64) * max(norm(hopping), 1.0)
        left, singular, right, discarded =
            tsvd(hopping, (1, 3), (2, 4); trunc=truncbelow(threshold))
        reconstructed = permute(left * singular * right, (1, 3), (2, 4))
        @test dim(domain(left)) == 2
        @test dim(codomain(right)) == 2
        @test discarded <= threshold
        @test norm(reconstructed - hopping) < 1e-12
    end
end
