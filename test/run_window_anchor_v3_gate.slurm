#!/usr/bin/env bash
#SBATCH --job-name=tk_anchor_v3_gate
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_v3_gate/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_v3_gate/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v3_20260919"
readonly converter_root="${migration_root}/converter_anchor_v2_20260918"
readonly source_root="${migration_root}/small_mpi_regression/source_d16"
readonly gate_root="${migration_root}/window_anchor_v3_gate/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_anchor_v3_gate_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_anchor_v3_gate_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -f "${source_root}/CURRENT"
test ! -e "${gate_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${gate_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model="${physical_label}"
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SWEEP_DIMS=16:16 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_VERIFY_WRITES=true PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_ANCHORS=true
export ENV_WINDOW_VERIFY_OWNERSHIP=true
export ENV_WINDOW_PROGRESS_SITES=1

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  start.jl test/run_window_anchor_v3_gate.slurm >"${gate_root}/engine_sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${gate_root}/converter_sha256.txt"
cp "${source_root}/CURRENT" "${gate_root}/source_CURRENT"

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib combined_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0
    test ! -e "${primary}" || primary_kib=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || spill_kib=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s\n' "$(date +%s)" "${primary_kib}" "${spill_kib}" \
      "$((primary_kib + spill_kib))" >>"${destination}"
    sleep 1
  done
}

current_generation() {
  printf '%s/%s' "$1" "$(tr -d '\n' <"$1/CURRENT")"
}

run_case() {
  local label=$1 nranks=$2 stable=$3 transient_gb=$4 reserve_gb=$5 scheduled=$6
  local output="${gate_root}/${label}_mps"
  local primary="${shm_parent}/${label}" spill="${local_parent}/${label}_spill"
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export ENV_PRIMARY_RESERVE_GB="${reserve_gb}"
  export ENV_STABLE_TERM_OWNERS="${stable}"
  export ENV_WINDOW_TRANSIENT_GB="${transient_gb}"
  if test "${nranks}" -eq 1; then
    export SVD_MODE=serial
  else
    export SVD_MODE=mpi
  fi
  export RUN_TAG="anchor_v3_${label}_${SLURM_JOB_ID}"
  if test "${scheduled}" = true; then
    export ENVIRONMENT_WINDOW_SITES=schedule
    export ENVIRONMENT_WINDOW_SCHEDULE=16:2,1000:32,5000:32,20000:16
  else
    export ENVIRONMENT_WINDOW_SITES=2
    unset ENVIRONMENT_WINDOW_SCHEDULE || true
  fi

  /usr/bin/time -v julia --project=. -e "
      using MPI
      launcher=MPI.mpiexec()
      child=\`julia --project=. start.jl\`
      run(\`\$launcher -n ${nranks} \$child\`)
  " >"${gate_root}/${label}.out" 2>"${gate_root}/${label}.err" &
  local run_pid=$! monitor_pid status=0
  monitor_storage "${primary}" "${spill}" "${run_pid}" "${gate_root}/${label}.storage.txt" &
  monitor_pid=$!
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || true
  test "${status}" -eq 0

  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  grep -Eq 'ENVIRONMENT_WINDOW_SELECTION D=16 window=2' "${gate_root}/${label}.out"
  for direction in forward reverse; do
    grep -Eq "WINDOW_WORK direction=${direction}.*window=2 anchors=true.*prepare_calls=3.*environment_site_updates=5.*anchor_snapshots=1.*anchor_restores=1" \
      "${gate_root}/${label}.out"
  done
  if awk -v x="${transient_gb}" 'BEGIN {exit !(x>0)}'; then
    grep -q 'WINDOW_TRANSIENT_STATS direction=right' "${gate_root}/${label}.out"
    grep -q 'WINDOW_TRANSIENT_STATS direction=left' "${gate_root}/${label}.out"
    ! grep -Eq 'Transient .* entries lack shared backing|Tensor file not found' \
      "${gate_root}/${label}.out" "${gate_root}/${label}.err"
  fi
  grep -Eq 'drops = [1-9][0-9]*' "${gate_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${gate_root}/${label}.out"
  local found_primary found_spill
  found_primary=$(awk -F= '$1=="ENVIRONMENT_STORE_ROOT" {print $2;exit}' "${gate_root}/${label}.out")
  found_spill=$(awk -F= '$1=="ENVIRONMENT_STORE_SPILL_ROOT" {print $2;exit}' "${gate_root}/${label}.out")
  test -n "${found_primary}" && test -n "${found_spill}"
  test ! -e "${found_primary}" && test ! -e "${found_spill}"
  local short_file="${engine_root}/out_short/${physical_label}/MPS_Lx3_Ly2_Ne6_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
  test -s "${short_file}"
  cp "${short_file}" "${gate_root}/${label}.out_short.txt"
}

# Reference keeps the v2 shared-backing ownership path.  Candidate cases test
# stable ownership with resident prepass at 1/2/4 ranks and near-zero capacity.
run_case legacy_r4 4 false 0 8 false
run_case resident_r1 1 true 0.25 8 true
run_case resident_r2 2 true 0.25 8 true
run_case resident_r4 4 true 0.25 8 true
run_case fallback_r4 4 true 0.0000001 1000000 true

compare_pair() {
  local candidate=$1
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${gate_root}/legacy_r4_mps")" \
    "$(current_generation "${gate_root}/${candidate}_mps")" \
    | tee "${gate_root}/legacy_vs_${candidate}.txt"
  awk -F= '$1=="normalized_overlap" {found=1;ok=($2>=1-1e-10)} END {exit !(found&&ok)}' \
    "${gate_root}/legacy_vs_${candidate}.txt"
  awk -F= '$1=="fidelity_per_site" {found=1;ok=($2>=1-1e-12)} END {exit !(found&&ok)}' \
    "${gate_root}/legacy_vs_${candidate}.txt"
  test "$(awk -F= '$1=="site_spaces_match" {print $2}' "${gate_root}/legacy_vs_${candidate}.txt")" = true
}

compare_rows() {
  local candidate=$1
  tr -d '[] ' <"${gate_root}/legacy_r4.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${gate_root}/legacy_r4.numeric"
  tr -d '[] ' <"${gate_root}/${candidate}.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${gate_root}/${candidate}.numeric"
  paste "${gate_root}/legacy_r4.numeric" "${gate_root}/${candidate}.numeric" | awk '{
    dt=$1-$3;if(dt<0)dt=-dt;de=$2-$4;if(de<0)de=-de;
    scale=($1<0?-$1:$1);other=($3<0?-$3:$3);if(other>scale)scale=other;
    tol=1e-12;if(1e-8*scale>tol)tol=1e-8*scale;
    if(dt>tol || de>1e-12)exit 1
  }'
}

for candidate in resident_r1 resident_r2 resident_r4 fallback_r4; do
  compare_pair "${candidate}"
  compare_rows "${candidate}"
done

test "$(awk 'NR>1 && $3>p {p=$3} END {print p+0}' "${gate_root}/fallback_r4.storage.txt")" -gt 4
printf 'window_anchor_v3_gate=PASS\n' | tee "${gate_root}/PASS"
