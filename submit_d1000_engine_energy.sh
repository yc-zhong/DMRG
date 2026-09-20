#!/usr/bin/env bash
set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly validation_root="${migration_root}/d1000_validation"
readonly batch_script="${migration_root}/engine_d1000_20260917/run_d1000_engine_energy.slurm"

mkdir -p "${validation_root}/logs"

submit_case() {
  local case_label=$1
  local ne=$2
  local dependency=${3:-}
  local start_dir="${validation_root}/${case_label}/itensor_converted"
  local output_dir="${validation_root}/${case_label}/engine_energy_r8_v2"
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args+=(--dependency="afterok:${dependency}")
  fi
  test -f "${start_dir}/conversion.toml"
  test ! -e "${output_dir}"
  sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name="tkE_D1k_${case_label}" \
    --output="${validation_root}/logs/%x.%j.out" \
    --error="${validation_root}/logs/%x.%j.err" \
    --export="ALL,CASE_LABEL=${case_label},NE=${ne},START_MPS_DIR=${start_dir},OUTPUT_MPS_DIR=${output_dir}" \
    "${batch_script}"
}

half_job=$(submit_case half 192 "${HALF_CONVERSION_JOB:-}")
doped_job=$(submit_case doped_n09375 180 "${DOPED_CONVERSION_JOB:-}")

printf 'half_energy_job=%s\ndoped_energy_job=%s\n' "${half_job}" "${doped_job}"
