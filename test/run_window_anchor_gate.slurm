#!/usr/bin/env bash
#SBATCH --job-name=tk_anchor_gate
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_gate_v2/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_gate_v2/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v2_20260918"
readonly converter_root="${migration_root}/converter_anchor_v2_20260918"
readonly source_root="${migration_root}/small_mpi_regression/source_d16"
readonly gate_root="${migration_root}/window_anchor_gate_v2/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_anchor_gate_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_anchor_gate_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge

cleanup() {
  rm -rf -- "${shm_parent}" "${local_parent}"
}
trap cleanup EXIT INT TERM

test -f "${source_root}/CURRENT"
test ! -e "${gate_root}"
mkdir -p "${gate_root}" "${shm_parent}" "${local_parent}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model="${physical_label}"
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=16:16 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=8 ENV_VERIFY_WRITES=true PROFILE_SEGMENTS=true

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  start.jl test/run_window_anchor_gate.slurm \
  >"${gate_root}/engine_sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${gate_root}/converter_sha256.txt"

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0
    test ! -e "${primary}" || primary_kib=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || spill_kib=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s\n' "$(date +%s)" "${primary_kib}" "${spill_kib}" >>"${destination}"
    sleep 1
  done
}

verify_cleanup() {
  local label=$1
  local output="${gate_root}/${label}.out"
  local primary spill
  primary=$(awk -F= '$1 == "ENVIRONMENT_STORE_ROOT" {print $2; exit}' "${output}")
  spill=$(awk -F= '$1 == "ENVIRONMENT_STORE_SPILL_ROOT" {print $2; exit}' "${output}")
  test -n "${primary}" && test -n "${spill}"
  test ! -e "${primary}" && test ! -e "${spill}"
}

run_case() {
  local label=$1 window=$2 anchors=$3 reserve_gb=$4
  local output="${gate_root}/${label}_mps"
  local primary="${shm_parent}/${label}" spill="${local_parent}/${label}_spill"
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export ENVIRONMENT_WINDOW_SITES="${window}"
  export ENVIRONMENT_WINDOW_ANCHORS="${anchors}"
  export ENV_PRIMARY_RESERVE_GB="${reserve_gb}"
  export RUN_TAG="anchor_gate_${label}_${SLURM_JOB_ID}"

  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n 4 $child`)
  ' >"${gate_root}/${label}.out" 2>"${gate_root}/${label}.err" &
  local run_pid=$!
  monitor_storage "${primary}" "${spill}" "${run_pid}" "${gate_root}/${label}.storage.txt" &
  local monitor_pid=$!
  local status=0
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || true
  test "${status}" -eq 0

  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  grep -q 'SEGMENT_TIMING.*name=sweep_total' "${gate_root}/${label}.out"
  grep -Eq 'drops = [1-9][0-9]*' "${gate_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${gate_root}/${label}.out"
  if test "${window}" -gt 0; then
    grep -q 'ENVIRONMENT_WINDOW direction=forward' "${gate_root}/${label}.out"
    grep -q 'ENVIRONMENT_WINDOW direction=reverse' "${gate_root}/${label}.out"
  fi
  verify_cleanup "${label}"

  local short_file="${engine_root}/out_short/${physical_label}/MPS_Lx3_Ly2_Ne6_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
  test -s "${short_file}"
  cp "${short_file}" "${gate_root}/${label}.out_short.txt"
}

run_case baseline 0 false 8
run_case legacy_w2 2 false 8
run_case anchor_w1 1 true 8
run_case anchor_w2 2 true 8
run_case anchor_w2_spill 2 true 1000000

grep -Eq 'WINDOW_WORK direction=forward.*window=2 anchors=false.*prepare_calls=3.*environment_site_updates=6' \
  "${gate_root}/legacy_w2.out"
grep -Eq 'WINDOW_WORK direction=reverse.*window=2 anchors=false.*prepare_calls=3.*environment_site_updates=6' \
  "${gate_root}/legacy_w2.out"
for direction in forward reverse; do
  grep -Eq "WINDOW_WORK direction=${direction}.*window=1 anchors=true.*prepare_calls=5.*environment_site_updates=4.*anchor_snapshots=3.*anchor_restores=3" \
    "${gate_root}/anchor_w1.out"
  grep -Eq "WINDOW_WORK direction=${direction}.*window=2 anchors=true.*prepare_calls=3.*environment_site_updates=5.*anchor_snapshots=1.*anchor_restores=1" \
    "${gate_root}/anchor_w2.out"
  grep -Eq "WINDOW_WORK direction=${direction}.*window=2 anchors=true.*prepare_calls=3.*environment_site_updates=5.*anchor_snapshots=1.*anchor_restores=1" \
    "${gate_root}/anchor_w2_spill.out"
done

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

compare_pair() {
  local label=$1 candidate=$2
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${gate_root}/baseline_mps")" \
    "$(current_generation "${gate_root}/${candidate}_mps")" \
    | tee "${gate_root}/${label}.txt"
  local overlap fidelity bra_norm ket_norm spaces
  overlap=$(awk -F= '$1 == "normalized_overlap" {print $2}' "${gate_root}/${label}.txt")
  fidelity=$(awk -F= '$1 == "fidelity_per_site" {print $2}' "${gate_root}/${label}.txt")
  bra_norm=$(awk -F= '$1 == "bra_norm2" {print $2}' "${gate_root}/${label}.txt")
  ket_norm=$(awk -F= '$1 == "ket_norm2" {print $2}' "${gate_root}/${label}.txt")
  spaces=$(awk -F= '$1 == "site_spaces_match" {print $2}' "${gate_root}/${label}.txt")
  awk -v value="${overlap}" 'BEGIN {exit !(value>=1.0-1.0e-11)}'
  awk -v value="${fidelity}" 'BEGIN {exit !(value>=1.0-1.0e-12)}'
  awk -v a="${bra_norm}" -v b="${ket_norm}" 'BEGIN {
    d=a-b; if(d<0)d=-d; exit !(d<=1.0e-10 && a>=1-1e-10 && a<=1+1e-10 && b>=1-1e-10 && b<=1+1e-10)
  }'
  test "${spaces}" = true
}

compare_sweep_rows() {
  local candidate=$1
  tr -d '[] ' <"${gate_root}/baseline.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${gate_root}/baseline.numeric"
  tr -d '[] ' <"${gate_root}/${candidate}.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${gate_root}/${candidate}.numeric"
  test -s "${gate_root}/baseline.numeric" && test -s "${gate_root}/${candidate}.numeric"
  test "$(wc -l <"${gate_root}/baseline.numeric")" -eq 2
  test "$(wc -l <"${gate_root}/${candidate}.numeric")" -eq 2
  paste "${gate_root}/baseline.numeric" "${gate_root}/${candidate}.numeric" | awk '{
    dt=$1-$3; if(dt<0)dt=-dt; de=$2-$4; if(de<0)de=-de;
    scale=($1<0?-$1:$1); other=($3<0?-$3:$3); if(other>scale)scale=other;
    tol=1e-12; if(1e-8*scale>tol)tol=1e-8*scale;
    if(dt>tol || de>1e-12) exit 1
  }'
}

for candidate in legacy_w2 anchor_w1 anchor_w2 anchor_w2_spill; do
  compare_pair "baseline_vs_${candidate}" "${candidate}"
  compare_sweep_rows "${candidate}"
done

for label in baseline legacy_w2 anchor_w1 anchor_w2 anchor_w2_spill; do
  awk 'NR>1 && $2>p {p=$2} END {print p+0}' "${gate_root}/${label}.storage.txt" \
    >"${gate_root}/${label}.primary_peak_kib"
  awk 'NR>1 && $3>p {p=$3} END {print p+0}' "${gate_root}/${label}.storage.txt" \
    >"${gate_root}/${label}.spill_peak_kib"
done
test "$(cat "${gate_root}/anchor_w2_spill.spill_peak_kib")" -gt 4

printf 'window_anchor_gate=PASS\n' | tee "${gate_root}/PASS"
