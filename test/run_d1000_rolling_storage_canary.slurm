#!/usr/bin/env bash
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_rolling_20260918"
readonly converter_root="${migration_root}/converter"
readonly source_root="${migration_root}/d1000_validation/half/engine_energy_r8_v2"
readonly canary_root="${migration_root}/rolling_d1000_canary_20260918/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_d1000_canary_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_d1000_canary_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge

cleanup() {
  rm -rf -- "${shm_parent}" "${local_parent}"
}
trap cleanup EXIT INT TERM

test -f "${source_root}/CURRENT"
test ! -e "${canary_root}"
mkdir -p "${canary_root}" "${shm_parent}" "${local_parent}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

export model="${physical_label}"
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=1000 INPUT_BOND_DIMENSION=1000 STOP_BOND_DIMENSION=1000
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=1000 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_VERIFY_WRITES=false

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl start.jl \
  >"${canary_root}/engine_sha256.txt"
df -h /dev/shm /tmp >"${canary_root}/node_storage_before.txt"
df -Pk /dev/shm /tmp >"${canary_root}/node_storage_before_kib.txt"

write_launch_manifest() {
  local label=$1
  local ranks=$2
  local primary=$3
  local spill=$4
  local reserve=$5
  local output=$6
  {
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'case=%s\ngeometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n' "${label}"
    printf 'U=12.0\nt=1.0\nty=1.0\npinning=left_edge_af\npinning_strength=0.1\n'
    printf 'Ne=192\nSz2=0\nD=1000\nsweep_dims=1000\nfirst_pass_reverse=false\n'
    printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff=1e-9\nsvd_mode=mpi\n'
    printf 'ranks=%s\nthreads_per_rank=1\nslurm_mem=96G\nslurm_time=16:00:00\n' "${ranks}"
    printf 'source=%s\noutput=%s\nprimary=%s\nspill=%s\nprimary_reserve_gb=%s\n' \
      "${source_root}" "${output}" "${primary}" "${spill}" "${reserve}"
    printf 'cache_gb_per_rank=0.125\nrolling_release=true\nenvironment_verify_writes=false\n'
  } >"${canary_root}/${label}.launch.txt"
}

monitor_storage() {
  local primary=$1
  local spill=$2
  local watched_pid=$3
  local destination=$4
  printf 'unix_time primary_kib spill_kib shm_used_kib shm_available_kib tmp_used_kib tmp_available_kib\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib spill_kib
    primary_kib=0
    spill_kib=0
    if test -e "${primary}"; then
      primary_kib=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    fi
    if test -e "${spill}"; then
      spill_kib=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    fi
    printf '%s %s %s ' "$(date +%s)" "${primary_kib}" "${spill_kib}" >>"${destination}"
    df -Pk /dev/shm | awk 'NR==2 {printf "%s %s ", $3, $4}' >>"${destination}"
    df -Pk /tmp | awk 'NR==2 {print $3, $4}' >>"${destination}"
    sleep 10
  done
}

verify_program_cleanup() {
  local label=$1
  local output="${canary_root}/${label}.out"
  local primary spill
  primary=$(awk -F= '$1 == "ENVIRONMENT_STORE_ROOT" {print $2; exit}' "${output}")
  spill=$(awk -F= '$1 == "ENVIRONMENT_STORE_SPILL_ROOT" {print $2; exit}' "${output}")
  test -n "${primary}"
  test -n "${spill}"
  test ! -e "${primary}"
  test ! -e "${spill}"
  printf 'program_cleanup_%s=PASS primary=%s spill=%s\n' \
    "${label}" "${primary}" "${spill}" >"${canary_root}/${label}.cleanup.txt"
}

run_case() {
  local label=$1
  local reserve_gb=$2
  local output="${canary_root}/${label}_mps"
  local primary="${shm_parent}/${label}"
  local spill="${local_parent}/${label}_spill"
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENV_PRIMARY_RESERVE_GB="${reserve_gb}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export RUN_TAG="d1000_canary_${label}_${SLURM_JOB_ID}"
  write_launch_manifest "${label}" 4 "${primary}" "${spill}" "${reserve_gb}" "${output}"

  cd "${engine_root}"
  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n 4 $child`)
  ' >"${canary_root}/${label}.out" 2>"${canary_root}/${label}.err" &
  local run_pid=$!
  monitor_storage "${primary}" "${spill}" "${run_pid}" \
    "${canary_root}/${label}.storage.txt" &
  local monitor_pid=$!
  local status=0
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || true
  test "${status}" -eq 0

  test -f "${output}/CURRENT"
  test -f "${output}/PREVIOUS"
  grep -Eq 'drops = [1-9][0-9]*' "${canary_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${canary_root}/${label}.out"
  verify_program_cleanup "${label}"

  local short_file="${engine_root}/out_short/${physical_label}/MPS_Lx32_Ly6_Ne192_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
  test -f "${short_file}"
  cp "${short_file}" "${canary_root}/${label}.out_short.txt"
}

run_case memory_primary 16
run_case forced_spill 1000

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

cd "${converter_root}"
julia --project=. compare_tensorkit_mps.jl \
  "$(current_generation "${canary_root}/memory_primary_mps")" \
  "$(current_generation "${canary_root}/forced_spill_mps")" \
  | tee "${canary_root}/memory_vs_spill.txt"

overlap=$(awk -F= '$1 == "normalized_overlap" {print $2}' "${canary_root}/memory_vs_spill.txt")
fidelity=$(awk -F= '$1 == "fidelity_per_site" {print $2}' "${canary_root}/memory_vs_spill.txt")
bra_norm=$(awk -F= '$1 == "bra_norm2" {print $2}' "${canary_root}/memory_vs_spill.txt")
ket_norm=$(awk -F= '$1 == "ket_norm2" {print $2}' "${canary_root}/memory_vs_spill.txt")
bra_dmax=$(awk -F= '$1 == "bra_max_bond_dimension" {print $2}' "${canary_root}/memory_vs_spill.txt")
ket_dmax=$(awk -F= '$1 == "ket_max_bond_dimension" {print $2}' "${canary_root}/memory_vs_spill.txt")
bra_blocks=$(awk -F= '$1 == "bra_max_block_count" {print $2}' "${canary_root}/memory_vs_spill.txt")
ket_blocks=$(awk -F= '$1 == "ket_max_block_count" {print $2}' "${canary_root}/memory_vs_spill.txt")

awk -v value="${overlap}" 'BEGIN {exit !(value >= 1.0-1.0e-10)}'
awk -v value="${fidelity}" 'BEGIN {exit !(value >= 1.0-1.0e-12)}'
awk -v bra="${bra_norm}" -v ket="${ket_norm}" 'BEGIN {
  difference=bra-ket; if (difference<0) difference=-difference
  exit !(difference<=1.0e-10 && bra>=1.0-1.0e-10 && bra<=1.0+1.0e-10 &&
    ket>=1.0-1.0e-10 && ket<=1.0+1.0e-10)
}'
test "${bra_dmax}" = "${ket_dmax}"
test "${bra_blocks}" = "${ket_blocks}"

extract_last_field() {
  local file=$1
  local field=$2
  tail -n 1 "${file}" | tr -d '[] ' | awk -F, -v field="${field}" '{print $field}'
}

memory_trunc=$(extract_last_field "${canary_root}/memory_primary.out_short.txt" 3)
spill_trunc=$(extract_last_field "${canary_root}/forced_spill.out_short.txt" 3)
memory_energy=$(extract_last_field "${canary_root}/memory_primary.out_short.txt" 4)
spill_energy=$(extract_last_field "${canary_root}/forced_spill.out_short.txt" 4)
awk -v a="${memory_energy}" -v b="${spill_energy}" 'BEGIN {
  difference=a-b; if (difference<0) difference=-difference
  exit !(difference<=1.0e-12)
}'
awk -v a="${memory_trunc}" -v b="${spill_trunc}" 'BEGIN {
  difference=a-b; if (difference<0) difference=-difference
  scale=(a<0 ? -a : a); other=(b<0 ? -b : b); if (other>scale) scale=other
  tolerance=1.0e-12; if (1.0e-8*scale>tolerance) tolerance=1.0e-8*scale
  exit !(difference<=tolerance)
}'

memory_primary_peak=$(awk 'NR>1 && $2>maximum {maximum=$2} END {print maximum+0}' \
  "${canary_root}/memory_primary.storage.txt")
memory_spill_peak=$(awk 'NR>1 && $3>maximum {maximum=$3} END {print maximum+0}' \
  "${canary_root}/memory_primary.storage.txt")
forced_spill_peak=$(awk 'NR>1 && $3>maximum {maximum=$3} END {print maximum+0}' \
  "${canary_root}/forced_spill.storage.txt")
tmp_min_available=$(awk 'NR>1 && (minimum==0 || $7<minimum) {minimum=$7} END {print minimum+0}' \
  "${canary_root}/forced_spill.storage.txt")

# Each store root contains a tiny ownership sentinel.  Allow that metadata, but
# require the forced-spill run to contain actual tensor payloads.
test "${memory_spill_peak}" -le 16
test "${forced_spill_peak}" -gt 16
awk -v peak="${memory_primary_peak}" 'BEGIN {
  extrapolated=25.0*peak
  floor=81169816.0
  if (floor>extrapolated) extrapolated=floor
  exit !(extrapolated<=95.0*1024.0*1024.0)
}'
awk -v peak="${forced_spill_peak}" -v available="${tmp_min_available}" 'BEGIN {
  exit !(25.0*peak<=0.70*available)
}'

{
  printf 'canary=PASS\n'
  printf 'memory_primary_peak_kib=%s\n' "${memory_primary_peak}"
  printf 'memory_spill_peak_kib=%s\n' "${memory_spill_peak}"
  printf 'forced_spill_peak_kib=%s\n' "${forced_spill_peak}"
  printf 'tmp_min_available_kib=%s\n' "${tmp_min_available}"
  printf 'memory_energy_per_site=%s\nspill_energy_per_site=%s\n' "${memory_energy}" "${spill_energy}"
  printf 'memory_truncation=%s\nspill_truncation=%s\n' "${memory_trunc}" "${spill_trunc}"
  printf 'normalized_overlap=%s\nfidelity_per_site=%s\n' "${overlap}" "${fidelity}"
} | tee "${canary_root}/PASS"

sacct -j "${SLURM_JOB_ID}" --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxDiskRead,MaxDiskWrite,ExitCode \
  -n -P >"${canary_root}/sacct_at_completion.txt" || true
