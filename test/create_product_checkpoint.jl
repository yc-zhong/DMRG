using LinearAlgebra
using MKL
using TensorOperations

length(ARGS) == 1 || error("usage: create_product_checkpoint.jl OUTPUT_DIRECTORY")
target = abspath(ARGS[1])
ispath(target) && error("Refusing to overwrite product checkpoint: $target")

BLAS.set_num_threads(parse(Int, get(ENV, "OMP_THREADS", "1")))
model_name = get(ENV, "model", "Square_Hubbard_U1_U1_OBC_LeftEdge")
model_file = joinpath(@__DIR__, "..", "model", model_name * ".jl")
isfile(model_file) || error("Model file not found: $model_file")
include(model_file)

tensors = TN_initial()
expected_sites = parameter["L"][1] * parameter["L"][2]
length(tensors) == expected_sites || error("Initial product-state length mismatch")

incoming = begin_empty_checkpoint_generation(target; generation="product-state")
for site in eachindex(tensors)
    tensor_save(tensors[site], joinpath(incoming, string(site)))
end
write_checkpoint_manifest(incoming, expected_sites;
    generation="product-state",
    metadata=Dict(
        "managed_by" => "DMRG-checkpoint-v2",
        "model" => parameter["model"],
        "Lx" => parameter["L"][1],
        "Ly" => parameter["L"][2],
        "Ne" => parameter["Ne"],
        "Sz2" => parameter["Sz2"],
        "U" => parameter["U"],
        "pinning" => parameter["pinning"],
        "left_edge_h" => parameter["left_edge_h"],
        "state" => "exact U(1)xU(1) product MPS",
    ))
commit_checkpoint_generation(target, incoming, expected_sites)
validate_checkpoint(target, expected_sites;
    physical_space=V[1], left_boundary=V_in[1], right_boundary=V_out[1])
println("PRODUCT_CHECKPOINT=", target)
println("PRODUCT_CHECKPOINT_SITES=", expected_sites)
println("PRODUCT_CHECKPOINT_NE=", parameter["Ne"])
println("PRODUCT_CHECKPOINT_SZ2=", parameter["Sz2"])
