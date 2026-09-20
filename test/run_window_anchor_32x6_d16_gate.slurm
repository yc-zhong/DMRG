#!/usr/bin/env bash
#SBATCH --job-name=tk_anchor_32d16
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_32x6_d16_gate/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_32x6_d16_gate/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v2_20260918"
readonly converter_root="${migration_root}/converter_anchor_v2_20260918"
readonly source_root="${migration_root}/window_environment_32x6_gate/run_12166595/source_half_product"
readonly gate_root="${migration_root}/window_anchor_32x6_d16_gate/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_anchor_32d16_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_anchor_32d16_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge

cleanup() { rm -rf -- "${shm_parent}" "${local_parent}"; }
trap cleanup EXIT INT TERM

test -f "${source_root}/CURRENT"
test ! -e "${gate_root}"
mkdir -p "${gate_root}" "${shm_parent}" "${local_parent}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model="${physical_label}"
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=16 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=8 ENV_VERIFY_WRITES=true PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=32

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  start.jl test/run_window_anchor_32x6_d16_gate.slurm >"${gate_root}/engine_sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" >"${gate_root}/converter_sha256.txt"
cp "${source_root}/CURRENT" "${gate_root}/source_CURRENT"

monitor_storage() {
  local primary=$1 spill=$2 pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib combined_kib\n' >"${destination}"
  while kill -0 "${pid}" 2>/dev/null; do
    local a=0 b=0
    test ! -e "${primary}" || a=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || b=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s\n' "$(date +%s)" "${a}" "${b}" "$((a+b))" >>"${destination}"
    sleep 1
  done
}

run_case() {
  local label=$1 anchors=$2
  local output="${gate_root}/${label}_mps"
  local primary="${shm_parent}/${label}" spill="${local_parent}/${label}_spill"
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export ENVIRONMENT_WINDOW_ANCHORS="${anchors}"
  export RUN_TAG="anchor_32d16_${label}_${SLURM_JOB_ID}"

  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n 4 $child`)
  ' >"${gate_root}/${label}.out" 2>"${gate_root}/${label}.err" &
  local run_pid=$! monitor_pid status=0
  monitor_storage "${primary}" "${spill}" "${run_pid}" "${gate_root}/${label}.storage.txt" &
  monitor_pid=$!
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || true
  test "${status}" -eq 0
  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  grep -Eq "WINDOW_WORK direction=forward.*window=32 anchors=${anchors}.*prepare_calls=6.*environment_site_updates=$(test "${anchors}" = true && printf 343 || printf 660).*anchor_snapshots=$(test "${anchors}" = true && printf 5 || printf 0).*anchor_restores=$(test "${anchors}" = true && printf 5 || printf 0)" "${gate_root}/${label}.out"
  grep -Eq 'drops = [1-9][0-9]*' "${gate_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${gate_root}/${label}.out"
  local found_primary found_spill
  found_primary=$(awk -F= '$1=="ENVIRONMENT_STORE_ROOT" {print $2;exit}' "${gate_root}/${label}.out")
  found_spill=$(awk -F= '$1=="ENVIRONMENT_STORE_SPILL_ROOT" {print $2;exit}' "${gate_root}/${label}.out")
  test -n "${found_primary}" && test -n "${found_spill}"
  test ! -e "${found_primary}" && test ! -e "${found_spill}"
  local short_file="${engine_root}/out_short/${physical_label}/MPS_Lx32_Ly6_Ne192_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
  test -s "${short_file}"
  cp "${short_file}" "${gate_root}/${label}.out_short.txt"
}

run_case legacy false
run_case anchor true

current_generation() {
  printf '%s/%s' "$1" "$(tr -d '\n' <"$1/CURRENT")"
}

cd "${converter_root}"
julia --project=. compare_tensorkit_mps.jl \
  "$(current_generation "${gate_root}/legacy_mps")" \
  "$(current_generation "${gate_root}/anchor_mps")" | tee "${gate_root}/legacy_vs_anchor.txt"
awk -F= '$1=="normalized_overlap" {found=1;ok=($2>=1-1e-10)} END {exit !(found&&ok)}' "${gate_root}/legacy_vs_anchor.txt"
awk -F= '$1=="fidelity_per_site" {found=1;ok=($2>=1-1e-12)} END {exit !(found&&ok)}' "${gate_root}/legacy_vs_anchor.txt"
test "$(awk -F= '$1=="site_spaces_match" {print $2}' "${gate_root}/legacy_vs_anchor.txt")" = true
a=$(awk -F= '$1=="bra_norm2" {print $2}' "${gate_root}/legacy_vs_anchor.txt")
b=$(awk -F= '$1=="ket_norm2" {print $2}' "${gate_root}/legacy_vs_anchor.txt")
awk -v a="${a}" -v b="${b}" 'BEGIN {d=a-b;if(d<0)d=-d;exit !(d<=1e-10 && a>=1-1e-10 && a<=1+1e-10 && b>=1-1e-10 && b<=1+1e-10)}'

for label in legacy anchor; do
  tr -d '[] ' <"${gate_root}/${label}.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${gate_root}/${label}.numeric"
done
paste "${gate_root}/legacy.numeric" "${gate_root}/anchor.numeric" | awk '{dt=$1-$3;if(dt<0)dt=-dt;de=$2-$4;if(de<0)de=-de;s=($1<0?-$1:$1);q=($3<0?-$3:$3);if(q>s)s=q;t=1e-12;if(1e-8*s>t)t=1e-8*s;exit !(dt<=t && de<=1e-12)}'

printf 'window_anchor_32x6_d16=PASS\n' | tee "${gate_root}/PASS"
