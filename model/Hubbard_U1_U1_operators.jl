module HubbardU1U1Operators

export HUBBARD_U1U1_CANONICAL_BASIS,
       HUBBARD_U1U1_LEGACY_BASIS,
       HUBBARD_U1U1_CANONICAL_FROM_LEGACY,
       hubbard_u1u1_legacy_operators,
       hubbard_u1u1_operators

const HUBBARD_U1U1_LEGACY_BASIS = ("Emp", "UpDn", "Up", "Dn")
const HUBBARD_U1U1_CANONICAL_BASIS = ("Emp", "Up", "Dn", "UpDn")

# If v_legacy is expressed in HUBBARD_U1U1_LEGACY_BASIS, then
# v_canonical = v_legacy[HUBBARD_U1U1_CANONICAL_FROM_LEGACY].
const HUBBARD_U1U1_CANONICAL_FROM_LEGACY = (1, 3, 4, 2)

function hubbard_u1u1_legacy_operators()
    return Dict{String,Matrix{Float64}}(
        "C_up" => [
            0 0 1 0
            0 0 0 0
            0 0 0 0
            0 1 0 0
        ],
        "C_dagup" => [
            0 0 0 0
            0 0 0 1
            1 0 0 0
            0 0 0 0
        ],
        "C_dn" => [
            0 0 0 1
            0 0 0 0
            0 1 0 0
            0 0 0 0
        ],
        "C_dagdn" => [
            0 0 0 0
            0 0 1 0
            0 0 0 0
            1 0 0 0
        ],
        "F" => [
            1 0 0 0
            0 1 0 0
            0 0 -1 0
            0 0 0 -1
        ],
        "Sz" => [
            0 0 0 0
            0 0 0 0
            0 0 0.5 0
            0 0 0 -0.5
        ],
        "S+" => [
            0 0 0 0
            0 0 0 0
            0 0 0 1
            0 0 0 0
        ],
        "S-" => [
            0 0 0 0
            0 0 0 0
            0 0 0 0
            0 0 1 0
        ],
        "Nupdn" => [
            0 0 0 0
            0 1 0 0
            0 0 0 0
            0 0 0 0
        ],
        "Nup" => [
            0 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 0
        ],
        "Ndn" => [
            0 0 0 0
            0 1 0 0
            0 0 0 0
            0 0 0 1
        ],
        "Nup_minus_Ndn" => [
            0 0 0 0
            0 0 0 0
            0 0 1 0
            0 0 0 -1
        ],
        "Ntol" => [
            0 0 0 0
            0 2 0 0
            0 0 1 0
            0 0 0 1
        ],
    )
end

"""
Return fresh local-operator matrices in TensorKit's canonical U(1) x U(1)
sector order `(Emp, Up, Dn, UpDn)`.

The supplied code encoded its dense matrices in
`(Emp, UpDn, Up, Dn)`. Applying the same exact permutation to both matrix
indices preserves the original operator while making its entries consistent
with the quantum-number labels attached to the TensorKit physical space.
"""
function hubbard_u1u1_operators()
    legacy = hubbard_u1u1_legacy_operators()
    p = collect(HUBBARD_U1U1_CANONICAL_FROM_LEGACY)
    return Dict(name => matrix[p, p] for (name, matrix) in legacy)
end

end
