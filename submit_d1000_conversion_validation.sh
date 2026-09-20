#!/usr/bin/env bash
set -euo pipefail

readonly cluster_root=/public4/home/sc56578/zyc/hubbard_flux
readonly migration_root="${cluster_root}/new_engine_migration"
readonly output_root="${migration_root}/d1000_validation"
readonly batch_script="${migration_root}/converter/run_d1000_conversion_validation.slurm"

mkdir -p "${output_root}/logs"

submit_case() {
  local case_label=$1
  local source_h5=$2
  local output_dir="${output_root}/${case_label}/itensor_converted"
  test ! -e "${output_dir}"
  sbatch --parsable \
    --job-name="tkconv_D1k_${case_label}" \
    --output="${output_root}/logs/%x.%j.out" \
    --error="${output_root}/logs/%x.%j.err" \
    --export="ALL,CASE_LABEL=${case_label},SOURCE_H5=${source_h5},OUTPUT_DIR=${output_dir}" \
    "${batch_script}"
}

half_source="${cluster_root}/hubbard_ladder_benchmark_output/mps_checkpoints/leftedgeaf_h010_L32_Ly6_D4000/half/mps_L32_Ly6_U12_half_leftedgeaf_h010_D1000_4000_D1000.h5"
doped_source="${cluster_root}/hubbard_ladder_benchmark_output/mps_checkpoints/leftedgeaf_h010_L32_Ly6_D4000/doped_n09375/mps_L32_Ly6_U12_doped_n09375_leftedgeaf_h010_D1000_4000_D1000.h5"

half_job=$(submit_case half "${half_source}")
doped_job=$(submit_case doped_n09375 "${doped_source}")

printf 'half_job=%s\ndoped_job=%s\n' "${half_job}" "${doped_job}"
