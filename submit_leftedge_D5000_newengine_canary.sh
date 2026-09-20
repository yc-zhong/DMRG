#!/usr/bin/env bash
set -euo pipefail

readonly cluster_root=/public4/home/sc56578/zyc/hubbard_flux
readonly migration_root="${cluster_root}/new_engine_migration"
readonly run_root="${migration_root}/leftedge_D5000_migration_20260917"
readonly convert_batch="${migration_root}/converter/run_leftedge_conversion.slurm"
readonly validate_batch="${migration_root}/converter/run_leftedge_conversion_validation.slurm"
readonly sweep_batch="${migration_root}/engine_leftedge_20260917/run_leftedge_fixedD_sweep.slurm"
readonly restart_gate_batch="${migration_root}/engine_leftedge_20260917/test/run_directional_restart_gate.slurm"

readonly half_source="${cluster_root}/hubbard_ladder_benchmark_output/mps_checkpoints/leftedge_continue_20260916/half/mps_L32_Ly6_U12_half_leftedgeaf_h010_D5000_continue_20260916_D5000.h5"
readonly doped_source="${cluster_root}/hubbard_ladder_benchmark_output/mps_checkpoints/leftedge_continue_20260916/doped_n09375/mps_L32_Ly6_U12_doped_n09375_leftedgeaf_h010_D5000_continue_20260916_D5000.h5"
readonly half_converted="${run_root}/half/itensor_converted"
readonly doped_converted="${run_root}/doped_n09375/itensor_converted"
readonly half_sweep="${run_root}/half/fixedD5000_forward_canary_r8"

test -f "${half_source}"
test -f "${doped_source}"
test ! -e "${half_converted}"
test ! -e "${doped_converted}"
test ! -e "${half_sweep}"
mkdir -p "${run_root}/logs"

submit_conversion() {
  local label=$1
  local source=$2
  local output=$3
  local dependency=${4:-}
  if [[ -n "${dependency}" ]]; then
    sbatch --parsable \
      --dependency="afterok:${dependency}" \
      --job-name="tkconv_D5k_${label}" \
      --output="${run_root}/logs/%x.%j.out" \
      --error="${run_root}/logs/%x.%j.err" \
      --export="ALL,CASE_LABEL=${label},SOURCE_H5=${source},OUTPUT_DIR=${output}" \
      "${convert_batch}"
    return
  fi
  sbatch --parsable \
    --job-name="tkconv_D5k_${label}" \
    --output="${run_root}/logs/%x.%j.out" \
    --error="${run_root}/logs/%x.%j.err" \
    --export="ALL,CASE_LABEL=${label},SOURCE_H5=${source},OUTPUT_DIR=${output}" \
    "${convert_batch}"
}

submit_validation() {
  local label=$1
  local source=$2
  local converted=$3
  local dependency=$4
  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --job-name="tkvalid_D5k_${label}" \
    --output="${run_root}/logs/%x.%j.out" \
    --error="${run_root}/logs/%x.%j.err" \
    --export="ALL,CASE_LABEL=${label},SOURCE_H5=${source},CONVERTED_DIR=${converted}" \
    "${validate_batch}"
}

half_convert_job=$(submit_conversion half "${half_source}" "${half_converted}")
half_validate_job=$(submit_validation half "${half_source}" "${half_converted}" "${half_convert_job}")
direction_gate_job=$(sbatch --parsable \
  --job-name=tk_direction_restart \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  "${restart_gate_batch}")
half_sweep_job=$(sbatch --parsable \
  --dependency="afterok:${half_validate_job}:${direction_gate_job}" \
  --job-name=tkD5k_half_forward \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=half,NE=192,BOND_D=5000,SWEEP_DIMS=5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${half_converted},OUTPUT_MPS_DIR=${half_sweep}" \
  "${sweep_batch}")

# Convert and validate doping only after the half conversion itself succeeds.
# Its first production sweep is intentionally not auto-submitted: the half
# canary's measured wall/RSS/I/O determines the safe rank and memory request.
doped_convert_job=$(submit_conversion doped_n09375 "${doped_source}" "${doped_converted}" "${half_convert_job}")
doped_validate_job=$(submit_validation doped_n09375 "${doped_source}" "${doped_converted}" "${doped_convert_job}")

printf 'half_convert_job=%s\n' "${half_convert_job}"
printf 'half_validate_job=%s\n' "${half_validate_job}"
printf 'direction_restart_gate_job=%s\n' "${direction_gate_job}"
printf 'half_forward_canary_job=%s\n' "${half_sweep_job}"
printf 'doped_convert_job=%s\n' "${doped_convert_job}"
printf 'doped_validate_job=%s\n' "${doped_validate_job}"
