using Test
using LinearAlgebra
using ITensors
using ITensorMPS
using TensorOperations

operator_source_candidates = (
    joinpath(@__DIR__, "..", "model", "Hubbard_U1_U1_operators.jl"),
    joinpath(@__DIR__, "..", "engine", "model", "Hubbard_U1_U1_operators.jl"),
)
operator_source_index = findfirst(isfile, operator_source_candidates)
isnothing(operator_source_index) && error("Cannot find Hubbard_U1_U1_operators.jl")
include(operator_source_candidates[operator_source_index])
using .HubbardU1U1Operators: hubbard_u1u1_operators

function dense_itensor_operator(name, site)
    return Array(op(name, site), prime(site), dag(site))
end

function code_hopping(operators)
    @tensor hopping_up[-1, -2, -3, -4] :=
        operators["C_dagup"][-1, 1] * operators["F"][1, -3] * operators["C_up"][-2, -4] -
        operators["C_up"][-1, 1] * operators["F"][1, -3] * operators["C_dagup"][-2, -4]
    @tensor hopping_dn[-1, -2, -3, -4] :=
        operators["C_dagdn"][-1, -3] * operators["F"][-2, 1] * operators["C_dn"][1, -4] -
        operators["C_dn"][-1, -3] * operators["F"][-2, 1] * operators["C_dagdn"][1, -4]
    return reshape(-(hopping_up + hopping_dn), 16, 16)
end

function itensor_hopping()
    sites = siteinds("Electron", 2; conserve_qns=false)
    os = OpSum()
    os += -1.0, "Cdagup", 1, "Cup", 2
    os += -1.0, "Cdagup", 2, "Cup", 1
    os += -1.0, "Cdagdn", 1, "Cdn", 2
    os += -1.0, "Cdagdn", 2, "Cdn", 1
    H = MPO(os, sites)
    tensor = H[1] * H[2]
    array = Array(
        tensor,
        prime(sites[1]),
        prime(sites[2]),
        dag(sites[1]),
        dag(sites[2]),
    )
    return reshape(array, 16, 16), sites[1]
end

@testset "TensorKit operator convention matches ITensor Electron" begin
    operators = hubbard_u1u1_operators()
    hopping_code = code_hopping(operators)
    hopping_itensor, site = itensor_hopping()

    @test operators["C_up"] == dense_itensor_operator("Cup", site)
    @test operators["C_dagup"] == dense_itensor_operator("Cdagup", site)
    @test operators["F"] == dense_itensor_operator("F", site)
    @test operators["Ntol"] == dense_itensor_operator("Ntot", site)
    @test operators["Nup"] == dense_itensor_operator("Nup", site)
    @test operators["Ndn"] == dense_itensor_operator("Ndn", site)
    @test operators["Nupdn"] == dense_itensor_operator("Nupdn", site)
    @test operators["Sz"] == dense_itensor_operator("Sz", site)

    # The supplied code represents the two spin species as commuting local
    # hard-core modes. Its existing F placement supplies the Jordan-Wigner
    # signs. Adding the internal up-mode parity recovers ITensor's local Cdn.
    nup = operators["C_dagup"] * operators["C_up"]
    up_parity = Matrix{Float64}(I, 4, 4) - 2nup
    physical_cdn = up_parity * operators["C_dn"]
    @test physical_cdn == dense_itensor_operator("Cdn", site)
    @test physical_cdn' == dense_itensor_operator("Cdagdn", site)

    @test hopping_code == hopping_itensor
    @test ishermitian(hopping_code)
end
