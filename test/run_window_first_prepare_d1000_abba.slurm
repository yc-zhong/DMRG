#!/usr/bin/env bash
#SBATCH --job-name=tk_firstwin_d1k
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_first_prepare_d1000_abba/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_first_prepare_d1000_abba/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v3_20260919"
readonly source_root="${migration_root}/d1000_validation/half/engine_energy_r8_v2"
readonly source_current="$(tr -d '\n' <"${source_root}/CURRENT")"
readonly source_generation="${source_root}/${source_current}"
readonly bench_root="${migration_root}/window_first_prepare_d1000_abba/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_firstwin_d1k_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_firstwin_d1k_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -d "${source_generation}"
test ! -e "${bench_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${bench_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model="${physical_label}"
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=1000 INPUT_BOND_DIMENSION=1000 STOP_BOND_DIMENSION=1000
export SVD_MODE=mpi KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export SOURCE_GENERATION="${source_generation}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=12 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=schedule
export ENVIRONMENT_WINDOW_SCHEDULE=1000:32,5000:32,9000:24,12000:20,20000:16
export ENVIRONMENT_WINDOW_ANCHORS=true
export ENV_WINDOW_VERIFY_OWNERSHIP=true
export ENV_WINDOW_PROGRESS_SITES=16

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  test/window_first_prepare_benchmark.jl test/run_window_first_prepare_d1000_abba.slurm \
  >"${bench_root}/engine_sha256.txt"
printf '%s\n' "${source_current}" >"${bench_root}/source_CURRENT"

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib combined_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local a=0 b=0
    test ! -e "${primary}" || a=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || b=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s\n' "$(date +%s)" "${a}" "${b}" "$((a+b))" >>"${destination}"
    sleep 2
  done
}

run_case() {
  local label=$1 stable=$2 transient_gb=$3
  local primary="${shm_parent}/${label}" spill="${local_parent}/${label}_spill"
  export RUN_TAG="firstwin_d1k_${label}_${SLURM_JOB_ID}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export ENV_STABLE_TERM_OWNERS="${stable}"
  export ENV_WINDOW_TRANSIENT_GB="${transient_gb}"

  local started finished pid monitor_pid status=0
  started=$(date +%s)
  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. test/window_first_prepare_benchmark.jl`
      run(`$launcher -n 4 $child`)
  ' >"${bench_root}/${label}.out" 2>"${bench_root}/${label}.err" &
  pid=$!
  monitor_storage "${primary}" "${spill}" "${pid}" "${bench_root}/${label}.storage.txt" &
  monitor_pid=$!
  wait "${pid}" || status=$?
  wait "${monitor_pid}" || true
  finished=$(date +%s)
  printf '%s\n' "$((finished-started))" >"${bench_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0
  grep -Eq 'FIRST_WINDOW_RESULT status=PASS.*bonds=1:32.*sites=192 D=1000 ranks=4.*environment_site_updates=190 anchor_snapshots=5' \
    "${bench_root}/${label}.out"
  grep -Eq 'ENVIRONMENT_WINDOW_SELECTION D=1000 window=32 configured=schedule' \
    "${bench_root}/${label}.out"
  grep -q 'SEGMENT_TIMING label=first-window-forward name=window_prepass_mini' \
    "${bench_root}/${label}.out"
  grep -q 'SEGMENT_TIMING label=first-window-forward name=window_prepass_combined' \
    "${bench_root}/${label}.out"
  grep -q 'SEGMENT_TIMING label=first-window-forward name=window_retained_publish' \
    "${bench_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*|Tensor file not found|BACKING_ERROR' \
    "${bench_root}/${label}.out" "${bench_root}/${label}.err"
  local actual_primary actual_spill
  actual_primary=$(grep -m1 '^ENVIRONMENT_STORE_ROOT=' "${bench_root}/${label}.out" | cut -d= -f2-)
  actual_spill=$(grep -m1 '^ENVIRONMENT_STORE_SPILL_ROOT=' "${bench_root}/${label}.out" | cut -d= -f2-)
  test -n "${actual_primary}" && test -n "${actual_spill}"
  test ! -e "${actual_primary}" && test ! -e "${actual_spill}"
}

# ABBA order controls page-cache and first-launch bias.  Both algorithms build
# exactly the same first W32 forward environment and five future anchors.
run_case legacy_a false 0
run_case resident_a true 8
run_case resident_b true 8
run_case legacy_b false 0

printf 'case algorithm wall_seconds first_window_seconds mini_max_seconds combined_max_seconds publish_max_seconds snapshot_max_seconds primary_peak_kib spill_peak_kib\n' \
  >"${bench_root}/performance.tsv"
for case_name in legacy_a resident_a resident_b legacy_b; do
  algorithm=legacy
  case "${case_name}" in resident_*) algorithm=resident;; esac
  output="${bench_root}/${case_name}.out"
  first=$(awk -F'max_rank_seconds=' '/FIRST_WINDOW_RESULT/ {split($2,a," ");print a[1];exit}' "${output}")
  segment() { awk -v name="$1" '$0~"SEGMENT_TIMING label=first-window-forward name="name" " {for(i=1;i<=NF;i++)if($i~/^max_rank_seconds=/){split($i,a,"=");print a[2];exit}}' "${output}"; }
  primary=$(awk 'NR>1&&$1~/^[0-9]+$/&&$2>p{p=$2}END{print p+0}' "${bench_root}/${case_name}.storage.txt")
  spill=$(awk 'NR>1&&$1~/^[0-9]+$/&&$3>p{p=$3}END{print p+0}' "${bench_root}/${case_name}.storage.txt")
  printf '%s %s %s %s %s %s %s %s %s %s\n' "${case_name}" "${algorithm}" \
    "$(cat "${bench_root}/${case_name}.case_wall_seconds")" "${first}" \
    "$(segment window_prepass_mini)" "$(segment window_prepass_combined)" \
    "$(segment window_retained_publish)" "$(segment window_anchor_snapshot)" \
    "${primary}" "${spill}" >>"${bench_root}/performance.tsv"
done

awk 'NR>1 {n[$2]++; for(i=3;i<=10;i++)s[$2,i]+=$i}
  END {for(a in n){printf "%s",a;for(i=3;i<=10;i++)printf " %.9f",s[a,i]/n[a];printf "\n"}}' \
  "${bench_root}/performance.tsv" | sort >"${bench_root}/performance_mean.tsv"
printf 'window_first_prepare_d1000_abba=PASS\n' | tee "${bench_root}/PASS"
