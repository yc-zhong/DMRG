using Test
using LinearAlgebra
using TensorOperations

ENV["model"] = get(ENV, "model", "Square_Hubbard_U1_U1_CBC")
ENV["L1"] = get(ENV, "L1", "2")
ENV["L2"] = get(ENV, "L2", "2")
ENV["U"] = get(ENV, "U", "12")
ENV["Ne"] = get(ENV, "Ne", "4")
ENV["t2"] = get(ENV, "t2", "0")
ENV["t3"] = get(ENV, "t3", "0")

include("../model/Square_Hubbard_U1_U1_CBC.jl")

@testset "2x2 U1xU1 Hubbard model construction" begin
    @test diag(convert(Array, TensorKit_matrix("Ntol"))) == [0.0, 1.0, 1.0, 2.0]
    @test diag(convert(Array, TensorKit_matrix("Nupdn"))) == [0.0, 0.0, 0.0, 1.0]

    hopping1_error = norm(
        permute(H_1[1] * H_1[2], (1, 3), (2, 4)) - TensorKit_matrix("Hopping1"),
    )
    hopping2_error = norm(
        permute(H_2[1] * H_2[2], (1, 3), (2, 4)) - TensorKit_matrix("Hopping2"),
    )
    @test hopping1_error < 1e-12
    @test hopping2_error < 1e-12
    @test length(terms_onsite) == 4
    @test !isempty(terms)

    println("hopping_split_errors=", (hopping1_error, hopping2_error))
end
