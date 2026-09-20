#!/usr/bin/env bash
set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly run_root="${migration_root}/leftedge_D5000_migration_20260917"
readonly sweep_batch="${migration_root}/engine_leftedge_20260917/run_leftedge_fixedD_sweep.slurm"

readonly half_validate_job=${HALF_VALIDATE_JOB:-12162988}
readonly doped_validate_job=${DOPED_VALIDATE_JOB:-12162992}
readonly half_converted="${run_root}/half/itensor_converted"
readonly doped_converted="${run_root}/doped_n09375/itensor_converted"
readonly half_output="${run_root}/half/fixedD5000_fullsweep1_contiguous_r8"
readonly doped_output="${run_root}/doped_n09375/fixedD5000_fullsweep1_contiguous_r8"

test -f "${sweep_batch}"
test ! -e "${half_output}"
test ! -e "${doped_output}"
mkdir -p "${run_root}/logs"

half_job=$(sbatch --parsable \
  --dependency="afterok:${half_validate_job}" \
  --job-name=tkD5k_half_full1 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=half,NE=192,BOND_D=5000,SWEEP_DIMS=5000:5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${half_converted},OUTPUT_MPS_DIR=${half_output}" \
  "${sweep_batch}")

doped_job=$(sbatch --parsable \
  --dependency="afterok:${doped_validate_job}" \
  --job-name=tkD5k_doped_full1 \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=doped_n09375,NE=180,BOND_D=5000,SWEEP_DIMS=5000:5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${doped_converted},OUTPUT_MPS_DIR=${doped_output}" \
  "${sweep_batch}")

printf 'half_contiguous_full_sweep_job=%s\n' "${half_job}"
printf 'doped_contiguous_full_sweep_job=%s\n' "${doped_job}"

