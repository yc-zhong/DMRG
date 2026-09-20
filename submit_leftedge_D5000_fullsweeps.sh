#!/usr/bin/env bash
set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly run_root="${migration_root}/leftedge_D5000_migration_20260917"
readonly sweep_batch="${migration_root}/engine_leftedge_20260917/run_leftedge_fixedD_sweep.slurm"

readonly restart_gate_job=${RESTART_GATE_JOB:-12162989}
readonly half_forward_job=${HALF_FORWARD_JOB:-12162990}
readonly doped_validate_job=${DOPED_VALIDATE_JOB:-12162992}

readonly half_forward="${run_root}/half/fixedD5000_forward_canary_r8"
readonly half_reverse="${run_root}/half/fixedD5000_fullsweep1_reverse_r8"
readonly doped_converted="${run_root}/doped_n09375/itensor_converted"
readonly doped_forward="${run_root}/doped_n09375/fixedD5000_fullsweep1_forward_r8"
readonly doped_reverse="${run_root}/doped_n09375/fixedD5000_fullsweep1_reverse_r8"

test -f "${sweep_batch}"
test ! -e "${half_reverse}"
test ! -e "${doped_forward}"
test ! -e "${doped_reverse}"
mkdir -p "${run_root}/logs"

half_reverse_job=$(sbatch --parsable \
  --dependency="afterok:${half_forward_job}" \
  --job-name=tkD5k_half_reverse \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=half,NE=192,BOND_D=5000,SWEEP_DIMS=5000,FIRST_PASS_REVERSE=true,START_MPS_DIR=${half_forward},OUTPUT_MPS_DIR=${half_reverse}" \
  "${sweep_batch}")

doped_forward_job=$(sbatch --parsable \
  --dependency="afterok:${doped_validate_job}:${restart_gate_job}" \
  --job-name=tkD5k_doped_forward \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=doped_n09375,NE=180,BOND_D=5000,SWEEP_DIMS=5000,FIRST_PASS_REVERSE=false,START_MPS_DIR=${doped_converted},OUTPUT_MPS_DIR=${doped_forward}" \
  "${sweep_batch}")

doped_reverse_job=$(sbatch --parsable \
  --dependency="afterok:${doped_forward_job}" \
  --job-name=tkD5k_doped_reverse \
  --output="${run_root}/logs/%x.%j.out" \
  --error="${run_root}/logs/%x.%j.err" \
  --export="ALL,CASE_LABEL=doped_n09375,NE=180,BOND_D=5000,SWEEP_DIMS=5000,FIRST_PASS_REVERSE=true,START_MPS_DIR=${doped_forward},OUTPUT_MPS_DIR=${doped_reverse}" \
  "${sweep_batch}")

printf 'half_reverse_job=%s\n' "${half_reverse_job}"
printf 'doped_forward_job=%s\n' "${doped_forward_job}"
printf 'doped_reverse_job=%s\n' "${doped_reverse_job}"

