#!/usr/bin/env bash
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_window_v3_20260918"
readonly converter_root="${migration_root}/converter"
readonly source_root="${migration_root}/d1000_validation/half/engine_energy_r8_v2"
readonly bench_root="${migration_root}/window_environment_d1000_benchmark/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_window_d1000_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_window_d1000_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge

cleanup() {
  rm -rf -- "${shm_parent}" "${local_parent}"
}
trap cleanup EXIT INT TERM

test -f "${source_root}/CURRENT"
test ! -e "${bench_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${bench_root}" "${shm_parent}" "${local_parent}"

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
export ENV_PRIMARY_RESERVE_GB=16 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl start.jl \
  >"${bench_root}/engine_sha256.txt"
cp "${source_root}/CURRENT" "${bench_root}/source_CURRENT"
df -h /dev/shm /tmp "${migration_root}" >"${bench_root}/storage_before.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${bench_root}/storage_before_kib.txt"
shm_available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4+0}')
tmp_available_kib=$(df -Pk /tmp | awk 'NR==2 {print $4+0}')
test "${shm_available_kib}" -ge 50331648
test "${tmp_available_kib}" -ge 67108864

write_launch_manifest() {
  local label=$1 window=$2 output=$3 primary=$4 spill=$5
  {
    printf 'job_id=%s\ncase=%s\n' "${SLURM_JOB_ID}" "${label}"
    printf 'geometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n'
    printf 'U=12.0\nt=1.0\nty=1.0\npinning=left_edge_af\npinning_strength=0.1\n'
    printf 'Ne=192\nSz2=0\nD=1000\nsweep_dims=1000\nfirst_pass_reverse=false\n'
    printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff=1e-9\nsvd_mode=mpi\n'
    printf 'ranks=4\nthreads_per_rank=1\nslurm_mem=128G\nslurm_time=06:00:00\n'
    printf 'environment_window_sites=%s\nrolling_release=true\ncache_gb_per_rank=0.125\n' "${window}"
    printf 'source=%s\nsource_current=%s\noutput=%s\nprimary=%s\nspill=%s\n' \
      "${source_root}" "$(tr -d '\n' <"${source_root}/CURRENT")" "${output}" "${primary}" "${spill}"
    printf 'primary_reserve_gb=16\nenvironment_verify_writes=false\nprofile_segments=true\n'
  } >"${bench_root}/${label}.launch.txt"
}

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib combined_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0
    test ! -e "${primary}" || primary_kib=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || spill_kib=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s\n' "$(date +%s)" "${primary_kib}" "${spill_kib}" \
      "$((primary_kib + spill_kib))" >>"${destination}"
    sleep 2
  done
}

verify_cleanup() {
  local label=$1
  local output="${bench_root}/${label}.out"
  local primary spill
  primary=$(awk -F= '$1 == "ENVIRONMENT_STORE_ROOT" {print $2; exit}' "${output}")
  spill=$(awk -F= '$1 == "ENVIRONMENT_STORE_SPILL_ROOT" {print $2; exit}' "${output}")
  test -n "${primary}" && test -n "${spill}"
  test ! -e "${primary}" && test ! -e "${spill}"
}

run_case() {
  local label=$1 window=$2
  local output="${bench_root}/${label}_mps"
  local primary="${shm_parent}/${label}" spill="${local_parent}/${label}_spill"
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export ENVIRONMENT_WINDOW_SITES="${window}"
  export RUN_TAG="window_d1000_${label}_${SLURM_JOB_ID}"
  write_launch_manifest "${label}" "${window}" "${output}" "${primary}" "${spill}"

  local case_started case_finished
  case_started=$(date +%s)
  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n 4 $child`)
  ' >"${bench_root}/${label}.out" 2>"${bench_root}/${label}.err" &
  local run_pid=$!
  monitor_storage "${primary}" "${spill}" "${run_pid}" "${bench_root}/${label}.storage.txt" &
  local monitor_pid=$!
  local status=0
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || true
  case_finished=$(date +%s)
  printf '%s\n' "$((case_finished - case_started))" >"${bench_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0

  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  for segment in initial_total hamiltonian_action krylov svd svd_lapack_active block_update \
      checkpoint_prepare checkpoint_site_read checkpoint_site_write \
      checkpoint_finalize sweep_total; do
    grep -q "SEGMENT_TIMING.*name=${segment}" "${bench_root}/${label}.out"
  done
  grep -q 'PROCESS_RSS.*sum_rank_high_water=' "${bench_root}/${label}.out"
  grep -Eq 'drops = [1-9][0-9]*' "${bench_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${bench_root}/${label}.out"
  if test "${window}" -gt 0; then
    grep -q 'ENVIRONMENT_WINDOW direction=forward' "${bench_root}/${label}.out"
  fi
  verify_cleanup "${label}"

  local short_file="${engine_root}/out_short/${physical_label}/MPS_Lx32_Ly6_Ne192_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
  test -f "${short_file}"
  cp "${short_file}" "${bench_root}/${label}.out_short.txt"
  tr -d '[] ' <"${bench_root}/${label}.out_short.txt" \
    >"${bench_root}/${label}.out_short.csv"
  awk -F, 'NR>1 {
    rows+=1
    if(NF!=6 || $1+0!=1 || $2+0!=1000) bad=1
  } END {exit !(rows==1 && !bad)}' "${bench_root}/${label}.out_short.csv"
  awk 'NR>1 && $2>p {p=$2} END {print p+0}' "${bench_root}/${label}.storage.txt" \
    >"${bench_root}/${label}.primary_peak_kib"
  awk 'NR>1 && $3>p {p=$3} END {print p+0}' "${bench_root}/${label}.storage.txt" \
    >"${bench_root}/${label}.spill_peak_kib"
  awk 'NR>1 && $4>p {p=$4} END {print p+0}' "${bench_root}/${label}.storage.txt" \
    >"${bench_root}/${label}.combined_peak_kib"
}

# Same allocation, same node and source.  The larger window runs first so a
# useful storage/speed comparison survives even if the tighter window is slow.
run_case baseline 0
run_case window32 32
run_case window16 16

segment_max_seconds() {
  local file=$1 label_prefix=$2 wanted_name=$3
  awk -v prefix="${label_prefix}" -v wanted="${wanted_name}" '
    $1 == "SEGMENT_TIMING" {
      label=""; name=""; value=""
      for(i=1;i<=NF;i++) {
        split($i,part,"=")
        if(part[1]=="label") label=part[2]
        if(part[1]=="name") name=part[2]
        if(part[1]=="max_rank_seconds") value=part[2]
      }
      if(index(label,prefix)==1 && name==wanted) found=value
    }
    END {if(found=="") exit 1; print found+0}
  ' "${file}"
}

rss_field() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    $1 == "PROCESS_RSS" && $2 ~ /^label=sweep-/ {
      for(i=1;i<=NF;i++) {
        split($i,part,"=")
        if(part[1]==wanted) found=part[2]
      }
    }
    END {if(found=="") exit 1; print found+0}
  ' "${file}"
}

printf 'case window case_wall_s initial_s sweep_s first_pass_profile_s window_prepare_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib\n' \
  >"${bench_root}/performance.tsv"
for specification in 'baseline 0' 'window32 32' 'window16 16'; do
  set -- ${specification}
  label=$1
  window=$2
  output="${bench_root}/${label}.out"
  initial_s=$(segment_max_seconds "${output}" initial- initial_total)
  sweep_s=$(segment_max_seconds "${output}" sweep- sweep_total)
  profile_total=$(awk -v a="${initial_s}" -v b="${sweep_s}" 'BEGIN {print a+b}')
  window_s=$(segment_max_seconds "${output}" sweep- window_prepare)
  h_action_s=$(segment_max_seconds "${output}" sweep- hamiltonian_action)
  krylov_s=$(segment_max_seconds "${output}" sweep- krylov)
  svd_s=$(segment_max_seconds "${output}" sweep- svd)
  block_s=$(segment_max_seconds "${output}" sweep- block_update)
  checkpoint_prepare_s=$(segment_max_seconds "${output}" sweep- checkpoint_prepare)
  checkpoint_finalize_s=$(segment_max_seconds "${output}" sweep- checkpoint_finalize)
  sum_rss=$(rss_field "${output}" sum_rank_high_water)
  max_rss=$(rss_field "${output}" max_rank_high_water)
  printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
    "${label}" "${window}" "$(cat "${bench_root}/${label}.case_wall_seconds")" \
    "${initial_s}" "${sweep_s}" "${profile_total}" "${window_s}" \
    "${h_action_s}" "${krylov_s}" "${svd_s}" "${block_s}" \
    "${checkpoint_prepare_s}" "${checkpoint_finalize_s}" "${sum_rss}" "${max_rss}" \
    >>"${bench_root}/performance.tsv"
done

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

compare_pair() {
  local label=$1 candidate=$2
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${bench_root}/baseline_mps")" \
    "$(current_generation "${bench_root}/${candidate}_mps")" \
    | tee "${bench_root}/${label}.txt"
  local overlap fidelity bra_norm ket_norm bra_dmax ket_dmax bra_blocks ket_blocks
  overlap=$(awk -F= '$1 == "normalized_overlap" {print $2}' "${bench_root}/${label}.txt")
  fidelity=$(awk -F= '$1 == "fidelity_per_site" {print $2}' "${bench_root}/${label}.txt")
  bra_norm=$(awk -F= '$1 == "bra_norm2" {print $2}' "${bench_root}/${label}.txt")
  ket_norm=$(awk -F= '$1 == "ket_norm2" {print $2}' "${bench_root}/${label}.txt")
  bra_dmax=$(awk -F= '$1 == "bra_max_bond_dimension" {print $2}' "${bench_root}/${label}.txt")
  ket_dmax=$(awk -F= '$1 == "ket_max_bond_dimension" {print $2}' "${bench_root}/${label}.txt")
  bra_blocks=$(awk -F= '$1 == "bra_max_block_count" {print $2}' "${bench_root}/${label}.txt")
  ket_blocks=$(awk -F= '$1 == "ket_max_block_count" {print $2}' "${bench_root}/${label}.txt")
  awk -v value="${overlap}" 'BEGIN {exit !(value>=1.0-1.0e-10)}'
  awk -v value="${fidelity}" 'BEGIN {exit !(value>=1.0-1.0e-12)}'
  awk -v a="${bra_norm}" -v b="${ket_norm}" 'BEGIN {
    d=a-b; if(d<0)d=-d; exit !(d<=1.0e-10 && a>=1-1e-10 && a<=1+1e-10 && b>=1-1e-10 && b<=1+1e-10)
  }'
  test "${bra_dmax}" = "${ket_dmax}"
  test "${bra_blocks}" = "${ket_blocks}"
}

compare_sweep_row() {
  local candidate=$1
  tr -d '[] ' <"${bench_root}/baseline.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${bench_root}/baseline.numeric"
  tr -d '[] ' <"${bench_root}/${candidate}.out_short.txt" | awk -F, 'NR>1 {print $3,$4}' >"${bench_root}/${candidate}.numeric"
  paste "${bench_root}/baseline.numeric" "${bench_root}/${candidate}.numeric" | awk '{
    dt=$1-$3; if(dt<0)dt=-dt; de=$2-$4; if(de<0)de=-de;
    scale=($1<0?-$1:$1); other=($3<0?-$3:$3); if(other>scale)scale=other;
    tol=1e-12; if(1e-8*scale>tol)tol=1e-8*scale;
    if(dt>tol || de>1e-12) exit 1
  }'
}

compare_pair baseline_vs_window32 window32
compare_pair baseline_vs_window16 window16
compare_sweep_row window32
compare_sweep_row window16

printf 'case primary_peak_kib spill_peak_kib combined_peak_kib projected_D5000_kib\n' \
  >"${bench_root}/capacity.tsv"
for label in baseline window32 window16; do
  primary=$(cat "${bench_root}/${label}.primary_peak_kib")
  spill=$(cat "${bench_root}/${label}.spill_peak_kib")
  combined=$(cat "${bench_root}/${label}.combined_peak_kib")
  printf '%s %s %s %s %s\n' "${label}" "${primary}" "${spill}" "${combined}" \
    "$((25 * combined))" >>"${bench_root}/capacity.tsv"
done

printf 'numerical=PASS\n' >"${bench_root}/NUMERICAL_PASS"
printf 'benchmark=COMPLETE\n' >"${bench_root}/BENCHMARK_COMPLETE"
