#!/usr/bin/env bash
set -euo pipefail

readonly half_validate_job=12164072
readonly doped_validate_job=12164073
readonly four_rank_gate_job=12164071
readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly run_root="${migration_root}/leftedge_D5000_migration_20260917"
readonly sweep_batch="${migration_root}/engine_leftedge_20260917/run_leftedge_fixedD_sweep.slurm"
readonly half_converted="${run_root}/half/itensor_converted"
readonly doped_converted="${run_root}/doped_n09375/itensor_converted"
readonly half_output="${run_root}/half/fixedD5000_fullsweep1_contiguous_r4_amd256"
readonly doped_output="${run_root}/doped_n09375/fixedD5000_fullsweep1_contiguous_r4_amd256"

test -f "${half_converted}/conversion.toml"
test -f "${doped_converted}/conversion.toml"
test ! -e "${half_output}"
test ! -e "${doped_output}"

half_sweep_job=$(sbatch --parsable \
  --partition=amd_256 --ntasks=4 --cpus-per-task=16 --mem=234G --time=48:00:00 \
  --dependency="afterok:${half_validate_job}:${four_rank_gate_job}" \
  --job-name=tkD5k256_half_full1 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=half,NE=192,BOND_D=5000,SWEEP_DIMS=5000:5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${half_converted},OUTPUT_MPS_DIR=${half_output}" \
  "${sweep_batch}")

doped_sweep_job=$(sbatch --parsable \
  --partition=amd_256 --ntasks=4 --cpus-per-task=16 --mem=234G --time=48:00:00 \
  --dependency="afterok:${doped_validate_job}:${four_rank_gate_job}" \
  --job-name=tkD5k256_doped_full1 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=doped_n09375,NE=180,BOND_D=5000,SWEEP_DIMS=5000:5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${doped_converted},OUTPUT_MPS_DIR=${doped_output}" \
  "${sweep_batch}")

printf 'half_sweep_amd256_job=%s\n' "${half_sweep_job}"
printf 'doped_sweep_amd256_job=%s\n' "${doped_sweep_job}"
