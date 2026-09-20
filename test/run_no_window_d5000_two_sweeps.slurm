#!/usr/bin/env bash
#SBATCH --partition=amd_1T
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=800G
#SBATCH --time=18:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/leftedge_D5000_no_window_2sweeps_20260919/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/leftedge_D5000_no_window_2sweeps_20260919/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v3_20260919"
readonly production_root="${migration_root}/leftedge_D5000_no_window_2sweeps_20260919"
readonly gate_pass="${NO_WINDOW_GATE_PASS:?NO_WINDOW_GATE_PASS is required}"
readonly case_name="${CASE:?CASE=half or CASE=doped is required}"

case "${case_name}" in
  half)
    readonly particle_number=192
    readonly source_root="${migration_root}/leftedge_D5000_migration_20260917/half/itensor_converted"
    ;;
  doped)
    readonly particle_number=180
    readonly source_root="${migration_root}/leftedge_D5000_migration_20260917/doped_n09375/itensor_converted"
    ;;
  *)
    echo "CASE must be half or doped, got ${case_name}" >&2
    exit 2
    ;;
esac

readonly run_root="${production_root}/${case_name}/job_${SLURM_JOB_ID}"
readonly output_root="${run_root}/mps"
readonly shm_parent="/dev/shm/${USER}/dmrg_D5k_nw2_${case_name}_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_D5k_nw2_${case_name}_${SLURM_JOB_ID}"
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test "$(cat "${gate_pass}")" = 'no_window_four_direction_gate=PASS'
test -f "${source_root}/1.jld2"
test -f "${source_root}/96.jld2"
test -f "${source_root}/192.jld2"
test -f "${source_root}/conversion.toml"
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

# The measured half-filled full environment peaked at 277.64 GiB.  Require
# enough RAM-backed and node-local capacity before importing or evolving MPS.
test "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" -ge 314572800
test "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" -ge 314572800

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=32 L2=6 Ne="${particle_number}" SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=5000 INPUT_BOND_DIMENSION=5000 STOP_BOND_DIMENSION=5000
# These are the calibrated new-engine definitions: three Lanczos vectors and
# squared discarded singular-value norm cutoff, not copied ITensor maxiter.
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=5000:5000:5000:5000
export FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export OUTPUT_MPS_DIR="${output_root}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=32 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0
export ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=false
export ENV_WINDOW_TRANSIENT_GB=0
export ENVIRONMENT_TMP_ROOT="${shm_parent}/environment"
export ENVIRONMENT_SPILL_ROOT="${local_parent}/environment_spill"
export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_D5k_nw2_${case_name}"
export RUN_TAG="D5k_nw2_${case_name}_${SLURM_JOB_ID}"

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  start.jl test/run_no_window_d5000_two_sweeps.slurm \
  >"${run_root}/engine_sha256.txt"
sha256sum "${source_root}/conversion.toml" "${source_root}/1.jld2" \
  "${source_root}/96.jld2" "${source_root}/192.jld2" \
  >"${run_root}/source_sha256_before.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f %s %T@\n' | sort \
  >"${run_root}/source_inventory_before.txt"
{
  printf 'job_id=%s\ncase=%s\ngeometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n' \
    "${SLURM_JOB_ID}" "${case_name}"
  printf 'U=12.0\nt=1.0\nty=1.0\nNe=%s\nSz2=0\npinning=left_edge_af\npinning_strength=0.1\n' \
    "${particle_number}"
  printf 'D=5000\nfull_sweeps=2\ndirectional_passes=4\ndirections=forward,reverse,forward,reverse\n'
  printf 'window_sites=0\nranks=4\nthreads_per_rank=1\nkrylov_dim=3\nkrylov_tol=1e-14\n'
  printf 'truncation_cutoff_squared_norm=1e-9\nsvd_mode=mpi\nsource=%s\noutput=%s\n' \
    "${source_root}" "${output_root}"
  printf 'gate_pass=%s\nslurm_mem=%s\nslurm_time_limit=%s\n' \
    "${gate_pass}" "${SLURM_MEM_PER_NODE:-800G}" "${SLURM_TIMELIMIT:-unknown}"
} >"${run_root}/launch_manifest.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"

monitor_storage() {
  local watched_pid=$1 destination=$2
  printf 'unix_time primary_kib spill_kib shm_available_kib tmp_available_kib project_available_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0
    test ! -e "${shm_parent}" || primary_kib=$(du -sk "${shm_parent}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${local_parent}" || spill_kib=$(du -sk "${local_parent}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s %s %s\n' "$(date +%s)" "${primary_kib}" "${spill_kib}" \
      "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" \
      "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" \
      "$(df -Pk "${migration_root}" | awk 'NR==2{print $4+0}')" >>"${destination}"
    sleep 30
  done
}

started=$(date +%s)
/usr/bin/time -v julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/engine.out" 2>"${run_root}/engine.err" &
pid=$!
monitor_storage "${pid}" "${run_root}/storage.tsv" &
monitor_pid=$!
status=0
wait "${pid}" || status=$?
wait "${monitor_pid}" || true
finished=$(date +%s)
printf '%s\n' "$((finished-started))" >"${run_root}/case_wall_seconds"
test "${status}" -eq 0

grep -Eq 'ENVIRONMENT_WINDOW_SELECTION D=5000 window=0 configured=0' "${run_root}/engine.out"
test "$(grep -c 'SEGMENT_TIMING label=initial-full-chain name=initial_total' "${run_root}/engine.out")" -eq 1
for spec in '1-forward' '2-reverse' '3-forward' '4-reverse'; do
  grep -q "SEGMENT_TIMING label=sweep-${spec} name=sweep_total" "${run_root}/engine.out"
done
test "$(grep -c 'SEGMENT_TIMING label=sweep-.* name=sweep_total' "${run_root}/engine.out")" -eq 4
test "$(grep -c 'PROCESS_RSS label=sweep-' "${run_root}/engine.out")" -eq 4
! grep -Eq 'Tensor file not found|SpaceMismatch|missing_drops = [1-9][0-9]*|BACKING_ERROR|OutOfMemoryError' \
  "${run_root}/engine.out" "${run_root}/engine.err"

test -f "${output_root}/CURRENT"
test -f "${output_root}/PREVIOUS"
for pointer in CURRENT PREVIOUS; do
  generation="${output_root}/$(tr -d '\n' < "${output_root}/${pointer}")"
  test -f "${generation}/checkpoint_manifest.toml"
  test "$(find "${generation}" -maxdepth 1 -type f \( -name '*.jld2' -o -name '*.bin' \) | wc -l | awk '{print $1}')" -eq 192
  grep -q 'complete = true' "${generation}/checkpoint_manifest.toml"
done
current="${output_root}/$(tr -d '\n' < "${output_root}/CURRENT")"
previous="${output_root}/$(tr -d '\n' < "${output_root}/PREVIOUS")"
grep -q 'reverse = true' "${current}/checkpoint_manifest.toml"
grep -q 'reverse = false' "${previous}/checkpoint_manifest.toml"

short_file="out_short/${model}/MPS_Lx32_Ly6_Ne${particle_number}_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
time_file="time/${model}/MPS_Lx32_Ly6_Ne${particle_number}_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
test -f "${short_file}" && test -f "${time_file}"
cp "${short_file}" "${run_root}/out_short.txt"
cp "${time_file}" "${run_root}/bond_timing.txt"
test "$(grep -c '^\[' "${run_root}/out_short.txt")" -eq 4

actual_primary=$(grep -m1 '^ENVIRONMENT_STORE_ROOT=' "${run_root}/engine.out" | cut -d= -f2-)
actual_spill=$(grep -m1 '^ENVIRONMENT_STORE_SPILL_ROOT=' "${run_root}/engine.out" | cut -d= -f2-)
test -n "${actual_primary}" && test -n "${actual_spill}"
test ! -e "${actual_primary}" && test ! -e "${actual_spill}"
sha256sum "${source_root}/conversion.toml" "${source_root}/1.jld2" \
  "${source_root}/96.jld2" "${source_root}/192.jld2" \
  >"${run_root}/source_sha256_after.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f %s %T@\n' | sort \
  >"${run_root}/source_inventory_after.txt"
cmp "${run_root}/source_sha256_before.txt" "${run_root}/source_sha256_after.txt"
cmp "${run_root}/source_inventory_before.txt" "${run_root}/source_inventory_after.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_after_kib.txt"
printf 'no_window_d5000_two_sweeps=PASS\ncase=%s\nfull_sweeps=2\ndirectional_passes=4\n' \
  "${case_name}" | tee "${run_root}/PASS"
