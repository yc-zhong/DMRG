#!/usr/bin/env bash
set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly run_root="${migration_root}/leftedge_D5000_migration_20260917"
readonly validate_batch="${migration_root}/converter/run_leftedge_conversion_validation.slurm"
readonly sweep_batch="${migration_root}/engine_leftedge_20260917/run_leftedge_fixedD_sweep.slurm"
readonly gate_batch="${migration_root}/engine_leftedge_20260917/test/run_contiguous_two_pass_gate.slurm"

readonly cluster_root=/public4/home/sc56578/zyc/hubbard_flux
readonly half_source="${cluster_root}/hubbard_ladder_benchmark_output/mps_checkpoints/leftedge_continue_20260916/half/mps_L32_Ly6_U12_half_leftedgeaf_h010_D5000_continue_20260916_D5000.h5"
readonly doped_source="${cluster_root}/hubbard_ladder_benchmark_output/mps_checkpoints/leftedge_continue_20260916/doped_n09375/mps_L32_Ly6_U12_doped_n09375_leftedgeaf_h010_D5000_continue_20260916_D5000.h5"
readonly half_converted="${run_root}/half/itensor_converted"
readonly doped_converted="${run_root}/doped_n09375/itensor_converted"
readonly half_output="${run_root}/half/fixedD5000_fullsweep1_contiguous_r4_amd256"
readonly doped_output="${run_root}/doped_n09375/fixedD5000_fullsweep1_contiguous_r4_amd256"

test -f "${half_converted}/conversion.toml"
test -f "${doped_converted}/conversion.toml"
test ! -e "${half_output}"
test ! -e "${doped_output}"
mkdir -p "${run_root}/logs"

gate_job=$(sbatch --parsable \
  --partition=amd_test --ntasks=4 --cpus-per-task=16 --mem=64G --time=00:30:00 \
  --job-name=tk_two_pass_gate_r4 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  "${gate_batch}")

half_validate_job=$(sbatch --parsable \
  --partition=amd_256 --ntasks=1 --cpus-per-task=64 --mem=220G --time=24:00:00 \
  --job-name=tkvalid256_D5k_half \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=half,SOURCE_H5=${half_source},CONVERTED_DIR=${half_converted}" \
  "${validate_batch}")

doped_validate_job=$(sbatch --parsable \
  --partition=amd_256 --ntasks=1 --cpus-per-task=64 --mem=220G --time=24:00:00 \
  --job-name=tkvalid256_D5k_doped \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=doped_n09375,SOURCE_H5=${doped_source},CONVERTED_DIR=${doped_converted}" \
  "${validate_batch}")

half_sweep_job=$(sbatch --parsable \
  --partition=amd_256 --ntasks=4 --cpus-per-task=16 --mem=234G --time=48:00:00 \
  --dependency="afterok:${half_validate_job}:${gate_job}" \
  --job-name=tkD5k256_half_full1 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=half,NE=192,BOND_D=5000,SWEEP_DIMS=5000:5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${half_converted},OUTPUT_MPS_DIR=${half_output}" \
  "${sweep_batch}")

doped_sweep_job=$(sbatch --parsable \
  --partition=amd_256 --ntasks=4 --cpus-per-task=16 --mem=234G --time=48:00:00 \
  --dependency="afterok:${doped_validate_job}:${gate_job}" \
  --job-name=tkD5k256_doped_full1 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=doped_n09375,NE=180,BOND_D=5000,SWEEP_DIMS=5000:5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${doped_converted},OUTPUT_MPS_DIR=${doped_output}" \
  "${sweep_batch}")

printf 'four_rank_gate_job=%s\n' "${gate_job}"
printf 'half_validate_amd256_job=%s\n' "${half_validate_job}"
printf 'doped_validate_amd256_job=%s\n' "${doped_validate_job}"
printf 'half_sweep_amd256_job=%s\n' "${half_sweep_job}"
printf 'doped_sweep_amd256_job=%s\n' "${doped_sweep_job}"
