#!/usr/bin/env bash
#SBATCH --job-name=tk_anchor_d1k
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_d1000_benchmark/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_d1000_benchmark/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v2_20260918"
readonly converter_root="${migration_root}/converter_anchor_v2_20260918"
readonly source_root="${migration_root}/d1000_validation/half/engine_energy_r8_v2"
source_current=$(tr -d '\n' <"${source_root}/CURRENT")
readonly source_generation="${source_root}/${source_current}"
readonly bench_root="${migration_root}/window_anchor_d1000_benchmark/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_anchor_d1000_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_anchor_d1000_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -f "${source_root}/CURRENT"
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
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=1000 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_generation}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=16 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=32

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  start.jl test/run_window_anchor_d1000_benchmark.slurm \
  >"${bench_root}/engine_sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${bench_root}/converter_sha256.txt"
printf '%s\n' "${source_current}" >"${bench_root}/source_CURRENT"
df -h /dev/shm /tmp "${migration_root}" >"${bench_root}/storage_before.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${bench_root}/storage_before_kib.txt"
test "$(df -Pk /dev/shm | awk 'NR==2 {print $4+0}')" -ge 50331648
test "$(df -Pk /tmp | awk 'NR==2 {print $4+0}')" -ge 67108864

write_launch_manifest() {
  local label=$1 anchors=$2 output=$3 primary=$4 spill=$5
  {
    printf 'job_id=%s\ncase=%s\n' "${SLURM_JOB_ID}" "${label}"
    printf 'geometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n'
    printf 'U=12.0\nt=1.0\nty=1.0\npinning=left_edge_af\npinning_strength=0.1\n'
    printf 'Ne=192\nSz2=0\nD=1000\nsweep_dims=1000\nfirst_pass_reverse=false\n'
    printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff=1e-9\nsvd_mode=mpi\n'
    printf 'ranks=4\nthreads_per_rank=1\nslurm_mem=128G\nslurm_time=03:00:00\n'
    printf 'environment_window_sites=32\nenvironment_window_anchors=%s\n' "${anchors}"
    printf 'rolling_release=true\ncache_gb_per_rank=0.125\nprimary_reserve_gb=16\n'
    printf 'source=%s\nsource_current=%s\noutput=%s\nprimary=%s\nspill=%s\n' \
      "${source_generation}" "${source_current}" \
      "${output}" "${primary}" "${spill}"
  } >"${bench_root}/${label}.launch.txt"
}

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  local valid_samples=0 failed_samples=0
  printf 'unix_time primary_kib spill_kib combined_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0 du_output
    if test -e "${primary}"; then
      if ! du_output=$(du -sk "${primary}" 2>/dev/null); then
        failed_samples=$((failed_samples + 1)); sleep 2; continue
      fi
      primary_kib=$(awk '{print $1+0}' <<<"${du_output}")
    fi
    if test -e "${spill}"; then
      if ! du_output=$(du -sk "${spill}" 2>/dev/null); then
        failed_samples=$((failed_samples + 1)); sleep 2; continue
      fi
      spill_kib=$(awk '{print $1+0}' <<<"${du_output}")
    fi
    printf '%s %s %s %s\n' "$(date +%s)" "${primary_kib}" "${spill_kib}" \
      "$((primary_kib + spill_kib))" >>"${destination}"
    valid_samples=$((valid_samples + 1))
    sleep 2
  done
  printf '# monitor_summary valid_samples=%s failed_samples=%s\n' \
    "${valid_samples}" "${failed_samples}" >>"${destination}"
}

verify_cleanup() {
  local label=$1 primary spill
  local output="${bench_root}/${label}.out"
  primary=$(awk -F= '$1 == "ENVIRONMENT_STORE_ROOT" {print $2; exit}' "${output}")
  spill=$(awk -F= '$1 == "ENVIRONMENT_STORE_SPILL_ROOT" {print $2; exit}' "${output}")
  test -n "${primary}" && test -n "${spill}"
  test ! -e "${primary}" && test ! -e "${spill}"
}

run_case() {
  local label=$1 anchors=$2
  local output="${bench_root}/${label}_mps"
  local primary="${shm_parent}/${label}" spill="${local_parent}/${label}_spill"
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${primary}"
  export ENVIRONMENT_SPILL_ROOT="${spill}"
  export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_${label}"
  export ENVIRONMENT_WINDOW_ANCHORS="${anchors}"
  export RUN_TAG="anchor_d1000_${label}_${SLURM_JOB_ID}"
  write_launch_manifest "${label}" "${anchors}" "${output}" "${primary}" "${spill}"

  local started finished run_pid monitor_pid status=0 monitor_status=0
  started=$(date +%s)
  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n 4 $child`)
  ' >"${bench_root}/${label}.out" 2>"${bench_root}/${label}.err" &
  run_pid=$!
  monitor_storage "${primary}" "${spill}" "${run_pid}" "${bench_root}/${label}.storage.txt" &
  monitor_pid=$!
  wait "${run_pid}" || status=$?
  wait "${monitor_pid}" || monitor_status=$?
  finished=$(date +%s)
  printf '%s\n' "$((finished - started))" >"${bench_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0
  test "${monitor_status}" -eq 0
  local valid_samples
  valid_samples=$(awk -F'[ =]' '/^# monitor_summary/ {for(i=1;i<=NF;i++)if($i=="valid_samples")print $(i+1)}' "${bench_root}/${label}.storage.txt")
  test -n "${valid_samples}" && test "${valid_samples}" -ge 10

  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  for segment in window_prepare window_anchor_snapshot window_anchor_restore \
      hamiltonian_action krylov svd block_update checkpoint_prepare \
      checkpoint_finalize sweep_total; do
    grep -q "SEGMENT_TIMING.*name=${segment}" "${bench_root}/${label}.out"
  done
  grep -q 'PROCESS_RSS.*sum_rank_high_water=' "${bench_root}/${label}.out"
  grep -Eq 'drops = [1-9][0-9]*' "${bench_root}/${label}.out"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${bench_root}/${label}.out"
  grep -Eq "WINDOW_WORK direction=forward.*window=32 anchors=${anchors}.*prepare_calls=6.*environment_site_updates=$(test "${anchors}" = true && printf 343 || printf 660).*anchor_snapshots=$(test "${anchors}" = true && printf 5 || printf 0).*anchor_restores=$(test "${anchors}" = true && printf 5 || printf 0)" "${bench_root}/${label}.out"
  verify_cleanup "${label}"

  local short_file="${engine_root}/out_short/${physical_label}/MPS_Lx32_Ly6_Ne192_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
  test -s "${short_file}"
  cp "${short_file}" "${bench_root}/${label}.out_short.txt"
  tr -d '[] ' <"${bench_root}/${label}.out_short.txt" >"${bench_root}/${label}.out_short.csv"
  awk -F, 'NR>1 {rows+=1; if(NF!=6 || $1+0!=1 || $2+0!=1000) bad=1}
    END {exit !(rows==1 && !bad)}' "${bench_root}/${label}.out_short.csv"
  for column in '2 primary' '3 spill' '4 combined'; do
    set -- ${column}
    awk -v col="$1" 'NR>1 && $1 ~ /^[0-9]+$/ && $col>p {p=$col} END {print p+0}' \
      "${bench_root}/${label}.storage.txt" >"${bench_root}/${label}.$2_peak_kib"
  done
}

# Same source, allocation, node, rank count and numerical settings.  ABBA order
# limits first-run and page-cache bias without mixing the two algorithms.
run_case legacy_a false
run_case anchor_a true
run_case anchor_b true
run_case legacy_b false

segment_seconds() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    $1=="SEGMENT_TIMING" && $2 ~ /^label=sweep-/ {
      name=""; value=""
      for(i=1;i<=NF;i++){split($i,p,"="); if(p[1]=="name")name=p[2]; if(p[1]=="max_rank_seconds")value=p[2]}
      if(name==wanted) found=value
    }
    END {if(found=="")exit 1; print found+0}' "${file}"
}

rss_field() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    $1=="PROCESS_RSS" && $2 ~ /^label=sweep-/ {
      for(i=1;i<=NF;i++){split($i,p,"="); if(p[1]==wanted)found=p[2]}
    }
    END {if(found=="")exit 1; print found+0}' "${file}"
}

printf 'case anchors case_wall_s sweep_s window_prepare_s anchor_snapshot_s anchor_restore_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib combined_peak_kib\n' >"${bench_root}/performance.tsv"
for spec in 'legacy_a false' 'anchor_a true' 'anchor_b true' 'legacy_b false'; do
  set -- ${spec}; label=$1; anchors=$2; output="${bench_root}/${label}.out"
  printf '%s %s %s' "${label}" "${anchors}" "$(cat "${bench_root}/${label}.case_wall_seconds")" >>"${bench_root}/performance.tsv"
  for segment in sweep_total window_prepare window_anchor_snapshot window_anchor_restore hamiltonian_action krylov svd block_update checkpoint_prepare checkpoint_finalize; do
    value=$(segment_seconds "${output}" "${segment}")
    printf ' %s' "${value}" >>"${bench_root}/performance.tsv"
  done
  sum_rss=$(rss_field "${output}" sum_rank_high_water)
  max_rss=$(rss_field "${output}" max_rank_high_water)
  printf ' %s %s %s %s %s\n' \
    "${sum_rss}" "${max_rss}" \
    "$(cat "${bench_root}/${label}.primary_peak_kib")" "$(cat "${bench_root}/${label}.spill_peak_kib")" \
    "$(cat "${bench_root}/${label}.combined_peak_kib")" >>"${bench_root}/performance.tsv"
done

awk '
  function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
  NR==1 {next}
  {
    if(NF!=18 || ($2!="true" && $2!="false")) bad=1
    for(i=3;i<=18;i++) if(!finite($i)) bad=1
    for(i=3;i<=15;i++) if(i!=6 && i!=7 && $i<=0) bad=1
    if($18<=0) bad=1
    count[$2]++
  }
  END {exit !(bad==0 && count["true"]==2 && count["false"]==2)}' \
  "${bench_root}/performance.tsv"

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

compare_pair() {
  local candidate=$1
  local result="${bench_root}/legacy_a_vs_${candidate}.txt"
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${bench_root}/legacy_a_mps")" \
    "$(current_generation "${bench_root}/${candidate}_mps")" | tee "${result}"
  awk -F= '
    function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
    $1=="normalized_overlap" {overlap=$2; seen_overlap=1}
    $1=="fidelity_per_site" {fidelity=$2; seen_fidelity=1}
    $1=="bra_norm2" {bra=$2; seen_bra=1}
    $1=="ket_norm2" {ket=$2; seen_ket=1}
    END {
      if(!(seen_overlap&&seen_fidelity&&seen_bra&&seen_ket)) exit 1
      if(!(finite(overlap)&&finite(fidelity)&&finite(bra)&&finite(ket))) exit 1
      if(overlap<1-1e-10 || overlap>1+1e-10) exit 1
      if(fidelity<1-1e-12 || fidelity>1+1e-12) exit 1
      d=bra-ket;if(d<0)d=-d
      exit !(d<=1e-10 && bra>=1-1e-10 && bra<=1+1e-10 && ket>=1-1e-10 && ket<=1+1e-10)
    }' "${result}"
  test "$(awk -F= '$1=="site_spaces_match" {print $2}' "${result}")" = true
}

for candidate in anchor_a anchor_b legacy_b; do
  compare_pair "${candidate}"
  paste <(awk -F, 'NR>1 {print $3,$4}' "${bench_root}/legacy_a.out_short.csv") \
        <(awk -F, 'NR>1 {print $3,$4}' "${bench_root}/${candidate}.out_short.csv") | \
    awk 'function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
      {seen+=1;if(NF!=4)bad=1;for(i=1;i<=4;i++)if(!finite($i))bad=1;dt=$1-$3;if(dt<0)dt=-dt;de=$2-$4;if(de<0)de=-de;s=($1<0?-$1:$1);q=($3<0?-$3:$3);if(q>s)s=q;t=1e-12;if(1e-8*s>t)t=1e-8*s;if(dt>t || de>1e-12)bad=1}
      END {exit !(seen==1 && bad==0)}'
done
printf 'numerical=PASS\n' >"${bench_root}/NUMERICAL_PASS"

awk 'NR>1 {kind=($2=="true"?"anchor":"legacy"); n[kind]++; for(i=3;i<=18;i++)s[kind,i]+=$i}
  END {print "kind case_wall_s sweep_s window_prepare_s anchor_snapshot_s anchor_restore_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib combined_peak_kib";
       for(k in n){printf "%s",k;for(i=3;i<=18;i++)printf " %.9g",s[k,i]/n[k];printf "\n"}}' \
  "${bench_root}/performance.tsv" >"${bench_root}/performance_mean.tsv"

awk '
  function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
  NR==1 {next}
  {if(NF!=17 || ($1!="anchor" && $1!="legacy"))bad=1;for(i=2;i<=17;i++)if(!finite($i))bad=1;count[$1]++}
  END {exit !(bad==0 && count["anchor"]==1 && count["legacy"]==1)}' \
  "${bench_root}/performance_mean.tsv"

metric() { awk -v kind=$1 -v col=$2 '$1==kind {print $col;found=1} END {if(!found)exit 1}' "${bench_root}/performance_mean.tsv"; }
max_metric() { awk -v anchors=$1 -v col=$2 'NR>1 && $2==anchors {if(!found || $col>value)value=$col;found=1} END {if(!found)exit 1;print value}' "${bench_root}/performance.tsv"; }
legacy_sweep=$(metric legacy 3); anchor_sweep=$(metric anchor 3)
legacy_env=$(metric legacy 4); anchor_env=$(metric anchor 4)
legacy_wall=$(metric legacy 2); anchor_wall=$(metric anchor 2)
awk -v old="${legacy_env}" -v new="${anchor_env}" 'BEGIN {exit !(new<=0.75*old)}'
awk -v old="${legacy_sweep}" -v new="${anchor_sweep}" 'BEGIN {exit !(new<=0.92*old)}'
awk -v old="${legacy_wall}" -v new="${anchor_wall}" 'BEGIN {exit !(new<=0.92*old)}'
for col in 7 8 9; do
  awk -v old="$(metric legacy "${col}")" -v new="$(metric anchor "${col}")" 'BEGIN {exit !(new<=1.05*old)}'
done
legacy_backing_max=$(max_metric false 18); anchor_backing_max=$(max_metric true 18)
legacy_rss_max=$(max_metric false 14); anchor_rss_max=$(max_metric true 14)
anchor_spill_max=$(max_metric true 17)
awk -v old="${legacy_backing_max}" -v new="${anchor_backing_max}" 'BEGIN {exit !(new<=1.25*old)}'
awk -v old="${legacy_rss_max}" -v new="${anchor_rss_max}" 'BEGIN {exit !(new<=1.10*old)}'
awk -v spill="${anchor_spill_max}" 'BEGIN {exit !(spill<=4)}'

{
  printf 'kind max_sum_rank_hwm_kib max_combined_backing_kib max_spill_kib\n'
  printf 'legacy %s %s %s\n' "${legacy_rss_max}" "${legacy_backing_max}" "$(max_metric false 17)"
  printf 'anchor %s %s %s\n' "${anchor_rss_max}" "${anchor_backing_max}" "${anchor_spill_max}"
} >"${bench_root}/capacity_max.tsv"

{
  printf 'metric legacy_mean anchor_mean anchor_over_legacy speedup\n'
  for spec in 'case_wall 2' 'sweep_total 3' 'window_prepare 4' 'h_action 7' 'krylov 8' 'svd 9' 'aggregate_rss_kib 13' 'combined_backing_kib 17'; do
    set -- ${spec}; old=$(metric legacy "$2"); new=$(metric anchor "$2")
    awk -v name="$1" -v old="${old}" -v new="${new}" 'BEGIN {printf "%s %.9g %.9g %.9g %.9g\n",name,old,new,new/old,old/new}'
  done
} >"${bench_root}/speedup.tsv"

printf 'performance=PASS\n' >"${bench_root}/PERFORMANCE_PASS"
printf 'benchmark=COMPLETE\n' >"${bench_root}/BENCHMARK_COMPLETE"
