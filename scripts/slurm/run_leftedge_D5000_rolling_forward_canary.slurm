#!/usr/bin/env bash
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=234G
#SBATCH --time=48:00:00

set -euo pipefail

: "${CANARY_PASS_FILE:?CANARY_PASS_FILE is required}"
: "${START_MPS_DIR:?START_MPS_DIR is required}"

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_rolling_20260918"
readonly run_parent="${migration_root}/leftedge_D5000_rolling_20260918/half"
readonly run_root="${run_parent}/forward_r4_job_${SLURM_JOB_ID}"
readonly output_mps="${run_root}/mps"
readonly shm_parent="/dev/shm/${USER}/dmrg_D5000_half_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_D5000_half_${SLURM_JOB_ID}"
readonly primary="${shm_parent}/environment"
readonly spill="${local_parent}/environment_spill"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge

cleanup() {
  rm -rf -- "${shm_parent}" "${local_parent}"
}
trap cleanup EXIT INT TERM

# This is deliberately a hard gate.  A queued job cannot bypass a failed or
# incomplete D1000 storage canary, nor can it run against changed engine code.
test -f "${CANARY_PASS_FILE}"
grep -qx 'canary=PASS' "${CANARY_PASS_FILE}"
readonly canary_dir="$(dirname "${CANARY_PASS_FILE}")"
test -f "${canary_dir}/engine_sha256.txt"
test -f "${START_MPS_DIR}/CURRENT"
test ! -e "${run_root}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"

cd "${engine_root}"
sha256sum -c "${canary_dir}/engine_sha256.txt" \
  | tee "${run_root}/engine_hash_check.txt"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl start.jl \
  >"${run_root}/engine_sha256.txt"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

export model="${physical_label}"
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=5000 INPUT_BOND_DIMENSION=5000 STOP_BOND_DIMENSION=5000
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=5000 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR
export OUTPUT_MPS_DIR="${output_mps}"
export ENVIRONMENT_TMP_ROOT="${primary}"
export ENVIRONMENT_SPILL_ROOT="${spill}"
export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_half_D5000_forward"
export ENV_PRIMARY_RESERVE_GB=16
export ENV_ROLLING_RELEASE=true
export ENV_LOCAL_CACHE_GB=0.125
export ENV_VERIFY_WRITES=false
export RUN_TAG="leftedge_half_D5000_rolling_forward_${SLURM_JOB_ID}"

source_generation=$(tr -d '\n' <"${START_MPS_DIR}/CURRENT")
test -d "${START_MPS_DIR}/${source_generation}"
{
  printf 'job_id=%s\ncase=half\ngeometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n' "${SLURM_JOB_ID}"
  printf 'U=12.0\nt=1.0\nty=1.0\npinning=left_edge_af\npinning_strength=0.1\n'
  printf 'Ne=192\nSz2=0\nD=5000\nsweep_dims=5000\nfirst_pass_reverse=false\n'
  printf 'purpose=one_forward_pass_resource_canary_not_full_bidirectional_sweep\n'
  printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff=1e-9\nsvd_mode=mpi\n'
  printf 'ranks=4\nthreads_per_rank=1\nslurm_cpus=64\nslurm_mem=234G\nslurm_time=48:00:00\n'
  printf 'start_mps=%s\nstart_generation=%s\noutput_mps=%s\n' \
    "${START_MPS_DIR}" "${source_generation}" "${output_mps}"
  printf 'canary_pass_file=%s\nprimary=%s\nspill=%s\nprimary_reserve_gb=16\n' \
    "${CANARY_PASS_FILE}" "${primary}" "${spill}"
  printf 'cache_gb_per_rank=0.125\nrolling_release=true\nenvironment_verify_writes=false\n'
} | tee "${run_root}/launch_manifest.txt"

df -h /dev/shm /tmp >"${run_root}/node_storage_before.txt"
df -Pk /dev/shm /tmp >"${run_root}/node_storage_before_kib.txt"

monitor_storage() {
  local watched_pid=$1
  local destination=$2
  printf 'unix_time primary_kib spill_kib shm_used_kib shm_available_kib tmp_used_kib tmp_available_kib\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0
    if test -e "${primary}"; then
      primary_kib=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    fi
    if test -e "${spill}"; then
      spill_kib=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    fi
    printf '%s %s %s ' "$(date +%s)" "${primary_kib}" "${spill_kib}" >>"${destination}"
    df -Pk /dev/shm | awk 'NR==2 {printf "%s %s ", $3, $4}' >>"${destination}"
    df -Pk /tmp | awk 'NR==2 {print $3, $4}' >>"${destination}"
    sleep 30
  done
}

/usr/bin/time -v julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/engine.out" 2>"${run_root}/engine.err" &
run_pid=$!
monitor_storage "${run_pid}" "${run_root}/storage.txt" &
monitor_pid=$!
status=0
wait "${run_pid}" || status=$?
wait "${monitor_pid}" || true
test "${status}" -eq 0

test -f "${output_mps}/CURRENT"
test -f "${output_mps}/PREVIOUS"
grep -Eq 'drops = [1-9][0-9]*' "${run_root}/engine.out"
! grep -Eq 'missing_drops = [1-9][0-9]*' "${run_root}/engine.out"

program_primary=$(awk -F= '$1 == "ENVIRONMENT_STORE_ROOT" {print $2; exit}' "${run_root}/engine.out")
program_spill=$(awk -F= '$1 == "ENVIRONMENT_STORE_SPILL_ROOT" {print $2; exit}' "${run_root}/engine.out")
test -n "${program_primary}"
test -n "${program_spill}"
test ! -e "${program_primary}"
test ! -e "${program_spill}"

short_file="${engine_root}/out_short/${physical_label}/MPS_Lx32_Ly6_Ne192_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
test -f "${short_file}"
cp "${short_file}" "${run_root}/out_short.txt"

primary_peak=$(awk 'NR>1 && $2>maximum {maximum=$2} END {print maximum+0}' "${run_root}/storage.txt")
spill_peak=$(awk 'NR>1 && $3>maximum {maximum=$3} END {print maximum+0}' "${run_root}/storage.txt")
{
  printf 'forward_resource_canary=PASS\n'
  printf 'primary_peak_kib=%s\nspill_peak_kib=%s\n' "${primary_peak}" "${spill_peak}"
  printf 'output_mps=%s\n' "${output_mps}"
  printf 'note=checkpoint_saved_but_full_bidirectional_D5000_sweep_not_yet_complete\n'
} | tee "${run_root}/PASS"
