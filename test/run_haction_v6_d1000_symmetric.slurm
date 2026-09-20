#!/usr/bin/env bash
#SBATCH --job-name=tk_hact_d1k_sym
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/haction_v6_d1000_symmetric/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/haction_v6_d1000_symmetric/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly baseline_engine=${migration_root}/engine_leftedge_haction_v2_20260919
readonly candidate_engine=${migration_root}/engine_leftedge_haction_v6_20260919
readonly converter_root=${migration_root}/converter_anchor_v2_20260918
readonly source_root=${migration_root}/d1000_validation/half/engine_energy_r8_v2
readonly d16_pass=${migration_root}/haction_v6_channels_d16_gate/run_12169755/PASS
readonly batch_script=${migration_root}/haction_v6_d1000_symmetric/run_haction_v6_d1000_symmetric.slurm
readonly direction_driver=${migration_root}/haction_v6_d1000_symmetric/single_direction_snapshot.jl
readonly run_root=${migration_root}/haction_v6_d1000_symmetric/run_${SLURM_JOB_ID}
readonly shm_parent=/dev/shm/${USER}/dmrg_hact_d1k_${SLURM_JOB_ID}
readonly local_parent=/tmp/${USER}/dmrg_hact_d1k_${SLURM_JOB_ID}
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test "$(cat "${d16_pass}")" = 'haction_v6_channels_d16_gate=PASS'
test -d "${baseline_engine}"
test -d "${candidate_engine}"
test -f "${source_root}/CURRENT"
source_current=$(tr -d '\n' <"${source_root}/CURRENT")
readonly source_current
readonly source_generation=${source_root}/${source_current}
test -d "${source_generation}"
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH=${migration_root}/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export TENSORKIT_TRANSFORMER_THREADS=1
export OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

export model=${physical_label}
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=1000 INPUT_BOND_DIMENSION=1000 STOP_BOND_DIMENSION=1000
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=1000 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR=${source_generation}
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=16 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0 ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=true ENV_WINDOW_TRANSIENT_GB=0
export ENV_TIERED_PINGPONG=false ENV_PRIMARY_CAP_GB=0
export H_ACTION_SCHEDULER=static_all_ranks
export MPI_TENSORMAP_REDUCE_MODE=packed
export DIRECTION_DRIVER=${direction_driver}

sha256sum "${candidate_engine}/MPS/TN.jl" "${candidate_engine}/MPS.jl" \
  "${candidate_engine}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl" \
  "${candidate_engine}/model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl" \
  "${candidate_engine}/start.jl" "${batch_script}" "${direction_driver}" \
  >"${run_root}/candidate_sha256.txt"
sha256sum "${baseline_engine}/MPS/TN.jl" "${baseline_engine}/MPS.jl" \
  "${baseline_engine}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl" \
  "${baseline_engine}/model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl" \
  "${baseline_engine}/start.jl" >"${run_root}/baseline_sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${run_root}/converter_sha256.txt"
printf '%s\n' "${source_current}" >"${run_root}/source_CURRENT"
find "${source_generation}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_before.tsv"
df -h /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"
test "$(df -Pk /dev/shm | awk 'NR==2 {print $4+0}')" -ge 33554432
test "$(df -Pk /tmp | awk 'NR==2 {print $4+0}')" -ge 33554432

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  local valid_samples=0 failed_samples=0
  printf 'unix_time primary_kib spill_kib combined_kib shm_available_kib tmp_available_kib\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0 du_output shm_available tmp_available
    if test -e "${primary}"; then
      if ! du_output=$(du -sk "${primary}" 2>/dev/null); then
        failed_samples=$((failed_samples + 1)); sleep 5; continue
      fi
      primary_kib=$(awk '{print $1+0}' <<<"${du_output}")
    fi
    if test -e "${spill}"; then
      if ! du_output=$(du -sk "${spill}" 2>/dev/null); then
        failed_samples=$((failed_samples + 1)); sleep 5; continue
      fi
      spill_kib=$(awk '{print $1+0}' <<<"${du_output}")
    fi
    shm_available=$(df -Pk /dev/shm | awk 'NR==2 {print $4+0}')
    tmp_available=$(df -Pk /tmp | awk 'NR==2 {print $4+0}')
    printf '%s %s %s %s %s %s\n' "$(date +%s)" "${primary_kib}" \
      "${spill_kib}" "$((primary_kib + spill_kib))" \
      "${shm_available}" "${tmp_available}" >>"${destination}"
    valid_samples=$((valid_samples + 1))
    sleep 5
  done
  printf '# monitor_summary valid_samples=%s failed_samples=%s\n' \
    "${valid_samples}" "${failed_samples}" >>"${destination}"
}

verify_cleanup() {
  local label=$1 output=${run_root}/$1.out primary spill
  primary=$(awk -F= '$1=="ENVIRONMENT_STORE_ROOT" {print $2;exit}' "${output}")
  spill=$(awk -F= '$1=="ENVIRONMENT_STORE_SPILL_ROOT" {print $2;exit}' "${output}")
  test -n "${primary}" && test -n "${spill}"
  test ! -e "${primary}" && test ! -e "${spill}"
  printf 'cleanup=PASS\nprimary=%s\nspill=%s\n' "${primary}" "${spill}" \
    >"${run_root}/${label}.cleanup.txt"
}

write_manifest() {
  local label=$1 engine=$2 mode=$3 pin_gb=$4 output=$5 primary=$6 spill=$7
  {
    printf 'job_id=%s\ncase=%s\n' "${SLURM_JOB_ID}" "${label}"
    printf 'geometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n'
    printf 'U=12.0\nt=1.0\nty=1.0\npinning=left_edge_af\npinning_strength=0.1\n'
    printf 'Ne=192\nSz2=0\nD=1000\ndirection=forward\n'
    printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff=1e-9\nsvd_mode=mpi\n'
    printf 'ranks=4\njulia_threads=1\ntensorkit_transformer_threads=1\nblas_threads=1\n'
    printf 'environment_window_sites=0\ntiered_pingpong=false\n'
    printf 'h_action_scheduler=static_all_ranks\nmpi_reduce_mode=packed\n'
    printf 'hopping_channel_mode=%s\nh_action_pin_cache_gb_per_rank=%s\n' "${mode}" "${pin_gb}"
    printf 'source=%s\nsource_current=%s\nengine=%s\n' \
      "${source_generation}" "${source_current}" "${engine}"
    printf 'output=%s\nprimary=%s\nspill=%s\n' "${output}" "${primary}" "${spill}"
  } >"${run_root}/${label}.launch.txt"
}

run_case() {
  local label=$1 engine=$2 mode=$3 pin_gb=$4
  local output=${run_root}/${label}_mps
  local primary=${shm_parent}/${label}
  local spill=${local_parent}/${label}_spill
  local work_root=${run_root}/${label}_work
  export ENGINE_ROOT=${engine}
  if test "${mode}" = baseline_full; then
    unset HOPPING_CHANNEL_MODE || true
  else
    export HOPPING_CHANNEL_MODE=${mode}
  fi
  export H_ACTION_PIN_CACHE_GB=${pin_gb}
  export OUTPUT_MPS_DIR=${output}
  export ENVIRONMENT_TMP_ROOT=${primary}
  export ENVIRONMENT_SPILL_ROOT=${spill}
  export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_${label}
  export RUN_TAG=hact_d1k_${label}_${SLURM_JOB_ID}
  mkdir -p "${work_root}/out/${model}" "${work_root}/out_short/${model}" \
    "${work_root}/time/${model}" "${work_root}/EE/${model}"
  write_manifest "${label}" "${engine}" "${mode}" "${pin_gb}" \
    "${output}" "${primary}" "${spill}"

  local started finished run_pid monitor_pid status=0 monitor_status=0
  started=$(date +%s)
  cd "${work_root}"
  /usr/bin/time -v julia --project="${engine}" -e '
      using MPI
      launcher=MPI.mpiexec()
      project=ENV["ENGINE_ROOT"]
      driver=ENV["DIRECTION_DRIVER"]
      child=`julia --project=$project $driver`
      run(`$launcher -n 4 $child`)
  ' >"${run_root}/${label}.out" 2>"${run_root}/${label}.err" &
  run_pid=$!
  monitor_storage "${primary}" "${spill}" "${run_pid}" \
    "${run_root}/${label}.storage.tsv" &
  monitor_pid=$!
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || monitor_status=$?
  finished=$(date +%s)
  printf '%s\n' "$((finished-started))" >"${run_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0
  test "${monitor_status}" -eq 0

  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  grep -q 'single_direction_snapshot=PASS direction=forward' "${run_root}/${label}.out"
  grep -q 'ENVIRONMENT_WINDOW_SELECTION D=1000 window=0' "${run_root}/${label}.out"
  ! grep -q '^WINDOW_WORK ' "${run_root}/${label}.out"
  grep -q 'SEGMENT_TIMING label=initial-full-chain name=initial_total' "${run_root}/${label}.out"
  for segment in hamiltonian_action krylov svd block_update checkpoint_prepare \
      checkpoint_finalize sweep_total; do
    grep -q "SEGMENT_TIMING label=sweep-.* name=${segment}" "${run_root}/${label}.out"
  done
  grep -q 'PROCESS_RSS label=sweep-.*sum_rank_high_water=' "${run_root}/${label}.out"
  grep -Eq 'drops = [1-9][0-9]*' "${run_root}/${label}.out"
  ! grep -Eq 'Tensor file not found|SpaceMismatch|Ambiguous environment backing|BACKING_ERROR' \
    "${run_root}/${label}.out" "${run_root}/${label}.err"
  if awk -v x="${pin_gb}" 'BEGIN {exit !(x>0)}'; then
    grep -q '^H_ACTION_PIN_CACHE rank=' "${run_root}/${label}.out"
    awk '
      $1=="H_ACTION_PIN_CACHE" && NF==12 {
        for(i=1;i<=NF;i++){split($i,p,"=");
          if(p[1]=="hits")hits+=p[2];
          if(p[1]=="admissions")admissions+=p[2];
          if(p[1]=="peak_bytes" && p[2]>peak)peak=p[2];
          if(p[1]=="rejections")rejections+=p[2]}
        records+=1
      }
      END {print "clean_records="records+0,"hits_lower_bound="hits+0,
        "admissions_lower_bound="admissions+0,"rejections_clean="rejections+0,
        "max_clean_rank_solve_peak_bytes="peak+0;
        exit !(records>0 && hits>0 && admissions>0)}' "${run_root}/${label}.out" \
      >"${run_root}/${label}.pin_summary.txt"
  fi
  verify_cleanup "${label}"

  # This driver calls MPS_sweep directly so that each case performs exactly
  # one direction.  Unlike the higher-level MPS()/MPS_SP() wrappers it does
  # not write out_short; the driver emits the same energy/truncation pair with
  # the validated checkpoint hash as a machine-readable result line.
  grep '^SINGLE_DIRECTION_RESULT direction=forward ' "${run_root}/${label}.out" \
    >"${run_root}/${label}.result.txt"
}

# Symmetric order gives every mode two same-source measurements while reducing
# first-launch, page-cache, and monotonic node-drift bias.
run_case baseline_a "${baseline_engine}" baseline_full 0
run_case split_a "${candidate_engine}" split_pruned 0
run_case fused_a "${candidate_engine}" fused_pruned 0
run_case fused_pin_a "${candidate_engine}" fused_pruned 2
run_case fused_pin_b "${candidate_engine}" fused_pruned 2
run_case fused_b "${candidate_engine}" fused_pruned 0
run_case split_b "${candidate_engine}" split_pruned 0
run_case baseline_b "${baseline_engine}" baseline_full 0

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

for candidate in split_a fused_a fused_pin_a fused_pin_b fused_b split_b baseline_b; do
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${run_root}/baseline_a_mps")" \
    "$(current_generation "${run_root}/${candidate}_mps")" \
    | tee "${run_root}/baseline_a_vs_${candidate}.txt"
  awk -F= '$1=="normalized_overlap" {found=1;ok=($2>=1-1e-10)} END {exit !(found&&ok)}' \
    "${run_root}/baseline_a_vs_${candidate}.txt"
  awk -F= '$1=="fidelity_per_site" {found=1;ok=($2>=1-1e-12)} END {exit !(found&&ok)}' \
    "${run_root}/baseline_a_vs_${candidate}.txt"
  test "$(awk -F= '$1=="site_spaces_match" {print $2}' \
    "${run_root}/baseline_a_vs_${candidate}.txt")" = true
done

result_field() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    /^SINGLE_DIRECTION_RESULT direction=forward / {
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]==wanted)found=p[2]}}
    END {if(found=="")exit 1;print found}' "${file}"
}

baseline_energy=$(result_field "${run_root}/baseline_a.result.txt" energy)
baseline_truncation=$(result_field "${run_root}/baseline_a.result.txt" truncation)
for candidate in split_a fused_a fused_pin_a fused_pin_b fused_b split_b baseline_b; do
  candidate_energy=$(result_field "${run_root}/${candidate}.result.txt" energy)
  candidate_truncation=$(result_field "${run_root}/${candidate}.result.txt" truncation)
  awk -v a="${baseline_energy}" -v b="${candidate_energy}" 'BEGIN {
    d=a-b;if(d<0)d=-d;exit !(d<=1e-9)}'
  awk -v a="${baseline_truncation}" -v b="${candidate_truncation}" 'BEGIN {
    d=a-b;if(d<0)d=-d;s=(a<0?-a:a);t=(b<0?-b:b);if(t>s)s=t;
    tol=1e-12;if(1e-8*s>tol)tol=1e-8*s;exit !(d<=tol)}'
done

segment_seconds() {
  local file=$1 label_pattern=$2 wanted=$3
  awk -v pattern="${label_pattern}" -v wanted="${wanted}" '
    $1=="SEGMENT_TIMING" && $2~pattern {
      name="";value=""
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="name")name=p[2];
        if(p[1]=="max_rank_seconds")value=p[2]}
      if(name==wanted)found=value
    }
    END {if(found=="")exit 1;print found+0}' "${file}"
}

rss_field() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    $1=="PROCESS_RSS" && $2~/^label=sweep-/ {
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]==wanted)found=p[2]}}
    END {if(found=="")exit 1;print found+0}' "${file}"
}

printf 'case mode pin_gb wall_s initial_s sweep_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib combined_peak_kib\n' \
  >"${run_root}/performance.tsv"
for spec in \
    'baseline_a baseline_full 0' 'split_a split_pruned 0' \
    'fused_a fused_pruned 0' 'fused_pin_a fused_pruned 2' \
    'fused_pin_b fused_pruned 2' 'fused_b fused_pruned 0' \
    'split_b split_pruned 0' 'baseline_b baseline_full 0'; do
  set -- ${spec}; label=$1; mode=$2; pin=$3; output=${run_root}/${label}.out
  initial=$(segment_seconds "${output}" '^label=initial-full-chain$' initial_total)
  sweep=$(segment_seconds "${output}" '^label=sweep-' sweep_total)
  haction=$(segment_seconds "${output}" '^label=sweep-' hamiltonian_action)
  krylov=$(segment_seconds "${output}" '^label=sweep-' krylov)
  svd=$(segment_seconds "${output}" '^label=sweep-' svd)
  block=$(segment_seconds "${output}" '^label=sweep-' block_update)
  cp_prepare=$(segment_seconds "${output}" '^label=sweep-' checkpoint_prepare)
  cp_finalize=$(segment_seconds "${output}" '^label=sweep-' checkpoint_finalize)
  sum_rss=$(rss_field "${output}" sum_rank_high_water)
  max_rss=$(rss_field "${output}" max_rank_high_water)
  primary_peak=$(awk 'NR>1 && $1~/^[0-9]+$/ && $2>p {p=$2} END {print p+0}' \
    "${run_root}/${label}.storage.tsv")
  spill_peak=$(awk 'NR>1 && $1~/^[0-9]+$/ && $3>p {p=$3} END {print p+0}' \
    "${run_root}/${label}.storage.tsv")
  combined_peak=$(awk 'NR>1 && $1~/^[0-9]+$/ && $4>p {p=$4} END {print p+0}' \
    "${run_root}/${label}.storage.tsv")
  printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
    "${label}" "${mode}" "${pin}" "$(cat "${run_root}/${label}.case_wall_seconds")" \
    "${initial}" "${sweep}" "${haction}" "${krylov}" "${svd}" "${block}" \
    "${cp_prepare}" "${cp_finalize}" "${sum_rss}" "${max_rss}" \
    "${primary_peak}" "${spill_peak}" "${combined_peak}" >>"${run_root}/performance.tsv"
done

awk '
  NR==1 {next}
  {key=$2 SUBSEP $3;count[key]++;mode[key]=$2;pin[key]=$3;
    for(i=4;i<=17;i++)sum[key,i]+=$i}
  END {
    print "mode pin_gb repeats mean_wall_s mean_initial_s mean_sweep_s mean_h_action_s mean_krylov_s mean_svd_s mean_block_update_s mean_checkpoint_prepare_s mean_checkpoint_finalize_s mean_sum_rank_hwm_kib mean_max_rank_hwm_kib mean_primary_peak_kib mean_spill_peak_kib mean_combined_peak_kib"
    for(key in count){printf "%s %s %d",mode[key],pin[key],count[key];
      for(i=4;i<=17;i++)printf " %.9g",sum[key,i]/count[key];printf "\n"}
  }' "${run_root}/performance.tsv" >"${run_root}/performance_mean.tsv"

find "${source_generation}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_after.tsv"
cmp "${run_root}/source_inventory_before.tsv" "${run_root}/source_inventory_after.tsv"
for label in baseline_a split_a fused_a fused_pin_a fused_pin_b fused_b split_b baseline_b; do
  test -z "$(find "${shm_parent}/${label}" -type f -print -quit 2>/dev/null)"
  test -z "$(find "${local_parent}/${label}_spill" -type f -print -quit 2>/dev/null)"
done

printf 'numerical=PASS\n' >"${run_root}/NUMERICAL_PASS"
printf 'benchmark=COMPLETE\n' >"${run_root}/BENCHMARK_COMPLETE"
printf 'haction_v6_d1000_symmetric=PASS\n' | tee "${run_root}/PASS"
