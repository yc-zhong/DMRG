using TensorKit

include("../MPS/TN.jl")
include("Hubbard_OBC_LeftEdge_U1_U1_core.jl")
using .HubbardOBCLeftEdgeU1U1

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
sanitize_run_tag(value) = replace(strip(value), r"[^A-Za-z0-9_.-]" => "_")

const _Lx = env_int("Lx", env_int("L1", 32))
const _Ly = env_int("L2", env_int("Ly", 6))
const _Ne = env_int("Ne", _Lx * _Ly)
const _Sz2 = env_int("SZ2", 0)
const _t = env_float("t", 1.0)
const _ty = env_float("ty", 1.0)
const _U = env_float("U", 12.0)
const _left_edge_h = env_float("PINNING_STRENGTH", env_float("left_edge_h", 0.1))
const _pinning = lowercase(get(ENV, "PINNING", "left_edge_af"))
const _hopping_channel_mode = Symbol(lowercase(strip(
    get(ENV, "HOPPING_CHANNEL_MODE", "split_pruned"),
)))
const _run_tag = sanitize_run_tag(get(ENV, "RUN_TAG", "default"))
isempty(_run_tag) && error("RUN_TAG must contain at least one safe character")

if !(_pinning == "left_edge_af" || (_pinning == "none" && iszero(_left_edge_h)))
    error("This adapter supports only PINNING=left_edge_af, or PINNING=none with zero strength")
end

const parameter = Dict{String,Any}(
    "model" => get(ENV, "model", "Square_Hubbard_U1_U1_OBC_LeftEdge"),
    "L" => [_Lx, _Ly],
    "U" => _U,
    "t" => _t,
    "ty" => _ty,
    "Ne" => _Ne,
    "Sz2" => _Sz2,
    "left_edge_h" => _left_edge_h,
    "pinning" => _pinning,
    "hopping_channel_mode" => String(_hopping_channel_mode),
    "run_tag" => _run_tag,
)

const _adapter = build_adapter(
    _Lx,
    _Ly;
    t=_t,
    ty=_ty,
    U=_U,
    left_edge_h=_left_edge_h,
    Ne=_Ne,
    Sz2=_Sz2,
    hopping_channel_mode=_hopping_channel_mode,
)

const out_put_file = joinpath(
    parameter["model"],
    "MPS_Lx$(_Lx)_Ly$(_Ly)_Ne$(_Ne)_Sz2$(_Sz2)_U$(_U)_t$(_t)_ty$(_ty)_leftedge$(_left_edge_h)_run$(_run_tag)",
)
const environment_tmp_root = abspath(get(ENV, "ENVIRONMENT_TMP_ROOT", "TMP_TN"))
const tmp_file = joinpath(environment_tmp_root, out_put_file)
configure_environment_store!(tmp_file)

const V = _adapter.V
const V_in = _adapter.V_in
const V_out = _adapter.V_out
const H_Hopping = _adapter.hopping_factors
const Ham_matrix = _adapter.Ham_matrix
const terms = _adapter.terms
const terms_onsite = _adapter.terms_onsite

Mapping(position::Vector{Int64}) = site_index(position[1], position[2], _Lx, _Ly)

function TensorKit_matrix(type::String; value::Float64=1.0)
    type == "Hopping1" && return value * _adapter.hopping_maps[1]
    type == "Hopping2" && return value * _adapter.hopping_maps[2]
    return tensor_operator(type; value=value, Vsite=_adapter.Vsite)
end

TN_initial() = initial_mps_tensors(_adapter)

if haskey(ENV, "SWEEP_DIMS")
    const SWEEP = [parse(Int, s) for s in strip.(split(ENV["SWEEP_DIMS"], r"[:,]")) if !isempty(s)]
end
