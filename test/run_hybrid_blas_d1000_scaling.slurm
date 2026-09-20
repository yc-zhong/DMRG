#!/usr/bin/env bash
#SBATCH --job-name=tk_d1k_blas
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_blas_d1000_scaling/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_blas_d1000_scaling/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root=${migration_root}/engine_leftedge_haction_v6_20260919
readonly converter_root=${migration_root}/converter_anchor_v2_20260918
readonly source_root=${migration_root}/d1000_validation/half/engine_energy_r8_v2
readonly d1000_pass=${migration_root}/haction_v6_d1000_symmetric/run_12171185/PASS
readonly affinity_pass=${migration_root}/hybrid_thread_affinity_smoke/run_12175622/PASS
readonly test_root=${migration_root}/hybrid_blas_d1000_scaling
readonly batch_script=${test_root}/run_hybrid_blas_d1000_scaling.slurm
readonly direction_driver=${test_root}/single_direction_snapshot.jl
readonly run_root=${test_root}/run_${SLURM_JOB_ID}
readonly shm_parent=/dev/shm/${USER}/dmrg_d1k_blas_${SLURM_JOB_ID}
readonly local_parent=/tmp/${USER}/dmrg_d1k_blas_${SLURM_JOB_ID}
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test "$(cat "${d1000_pass}")" = 'haction_v6_d1000_symmetric=PASS'
test "$(cat "${affinity_pass}")" = 'hybrid_thread_affinity_smoke=PASS'
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
export JULIA_NUM_THREADS=1 TENSORKIT_TRANSFORMER_THREADS=1
export OMP_PLACES=cores OMP_PROC_BIND=close MKL_DYNAMIC=false

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
export HOPPING_CHANNEL_MODE=split_pruned H_ACTION_PIN_CACHE_GB=0
export H_ACTION_SCHEDULER=static_all_ranks MPI_TENSORMAP_REDUCE_MODE=packed
export ENGINE_ROOT=${engine_root} DIRECTION_DRIVER=${direction_driver}

sha256sum "${engine_root}/MPS/TN.jl" "${engine_root}/MPS.jl" \
  "${engine_root}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl" \
  "${engine_root}/model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl" \
  "${batch_script}" "${direction_driver}" >"${run_root}/sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${run_root}/converter_sha256.txt"
printf '%s\n' "${source_current}" >"${run_root}/source_CURRENT"
find "${source_generation}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_before.tsv"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"
test "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" -ge 33554432
test "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" -ge 33554432

monitor_storage() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib combined_kib shm_available_kib tmp_available_kib\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local p=0 s=0
    test ! -e "${primary}" || p=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || s=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s %s %s\n' "$(date +%s)" "${p}" "${s}" "$((p+s))" \
      "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" \
      "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" >>"${destination}"
    sleep 5
  done
}

run_case() {
  local label=$1 threads=$2
  local output=${run_root}/${label}_mps
  local primary=${shm_parent}/${label}
  local spill=${local_parent}/${label}_spill
  local work=${run_root}/${label}_work
  export OMP_THREADS=${threads} OMP_NUM_THREADS=${threads}
  export MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}
  export OUTPUT_MPS_DIR=${output}
  export ENVIRONMENT_TMP_ROOT=${primary}
  export ENVIRONMENT_SPILL_ROOT=${spill}
  export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_${label}
  export RUN_TAG=d1k_${label}_${SLURM_JOB_ID}
  mkdir -p "${work}/out/${model}" "${work}/out_short/${model}" \
    "${work}/time/${model}" "${work}/EE/${model}"
  {
    printf 'case=%s\nblas_threads=%s\nranks=4\ncpus_per_rank=16\n' "${label}" "${threads}"
    printf 'julia_threads=1\ntransformer_threads=1\nsource=%s\noutput=%s\n' \
      "${source_generation}" "${output}"
  } >"${run_root}/${label}.launch.txt"

  local started pid monitor_pid status=0
  started=$(date +%s)
  cd "${work}"
  srun --exclusive --mpi=pmi2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores \
    julia --project="${engine_root}" "${direction_driver}" \
    >"${run_root}/${label}.out" 2>"${run_root}/${label}.err" &
  pid=$!
  monitor_storage "${primary}" "${spill}" "${pid}" \
    "${run_root}/${label}.storage.tsv" &
  monitor_pid=$!
  wait "${pid}" || status=$?
  wait "${monitor_pid}" || true
  printf '%s\n' "$(( $(date +%s) - started ))" >"${run_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0

  grep -q "THREAD_CONFIGURATION julia=1 tensor_transformer=1 blas=${threads}" \
    "${run_root}/${label}.out"
  grep -q 'single_direction_snapshot=PASS direction=forward' "${run_root}/${label}.out"
  grep -q 'ENVIRONMENT_WINDOW_SELECTION D=1000 window=0' "${run_root}/${label}.out"
  grep -q 'SEGMENT_TIMING label=initial-full-chain name=initial_total' "${run_root}/${label}.out"
  for segment in hamiltonian_action krylov svd block_update checkpoint_prepare \
      checkpoint_finalize sweep_total; do
    grep -q "SEGMENT_TIMING label=sweep-.* name=${segment}" "${run_root}/${label}.out"
  done
  ! grep -Eq 'Tensor file not found|SpaceMismatch|BACKING_ERROR|OutOfMemoryError' \
    "${run_root}/${label}.out" "${run_root}/${label}.err"
  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  grep '^SINGLE_DIRECTION_RESULT direction=forward ' "${run_root}/${label}.out" \
    >"${run_root}/${label}.result.txt"
  actual_primary=$(awk -F= '$1=="ENVIRONMENT_STORE_ROOT"{print $2;exit}' \
    "${run_root}/${label}.out")
  actual_spill=$(awk -F= '$1=="ENVIRONMENT_STORE_SPILL_ROOT"{print $2;exit}' \
    "${run_root}/${label}.out")
  test -n "${actual_primary}" && test -n "${actual_spill}"
  test ! -e "${actual_primary}" && test ! -e "${actual_spill}"
}

# Bracket the monotonic thread sweep with two one-thread baselines so a warm
# page cache or node drift cannot masquerade as a 16-thread speedup.
run_case blas1_a 1
run_case blas4 4
run_case blas8 8
run_case blas16 16
run_case blas1_b 1

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

for candidate in blas4 blas8 blas16 blas1_b; do
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${run_root}/blas1_a_mps")" \
    "$(current_generation "${run_root}/${candidate}_mps")" \
    >"${run_root}/blas1_a_vs_${candidate}.txt"
  awk -F= '$1=="normalized_overlap"{found=1;ok=($2>=1-1e-10)}END{exit !(found&&ok)}' \
    "${run_root}/blas1_a_vs_${candidate}.txt"
  awk -F= '$1=="fidelity_per_site"{found=1;ok=($2>=1-1e-12)}END{exit !(found&&ok)}' \
    "${run_root}/blas1_a_vs_${candidate}.txt"
  test "$(awk -F= '$1=="site_spaces_match"{print $2}' \
    "${run_root}/blas1_a_vs_${candidate}.txt")" = true
  baseline_energy=$(awk '{for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="energy")print p[2]}}' \
    "${run_root}/blas1_a.result.txt")
  candidate_energy=$(awk '{for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="energy")print p[2]}}' \
    "${run_root}/${candidate}.result.txt")
  baseline_trunc=$(awk '{for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="truncation")print p[2]}}' \
    "${run_root}/blas1_a.result.txt")
  candidate_trunc=$(awk '{for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="truncation")print p[2]}}' \
    "${run_root}/${candidate}.result.txt")
  awk -v a="${baseline_energy}" -v b="${candidate_energy}" \
    'BEGIN{d=a-b;if(d<0)d=-d;exit !(d<=1e-9)}'
  awk -v a="${baseline_trunc}" -v b="${candidate_trunc}" \
    'BEGIN{d=a-b;if(d<0)d=-d;s=(a<0?-a:a);t=(b<0?-b:b);if(t>s)s=t;
      tol=1e-12;if(1e-8*s>tol)tol=1e-8*s;exit !(d<=tol)}'
done

segment_seconds() {
  local file=$1 pattern=$2 wanted=$3
  awk -v pattern="${pattern}" -v wanted="${wanted}" '
    $1=="SEGMENT_TIMING" && $2~pattern {
      name="";value=""
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="name")name=p[2];
        if(p[1]=="max_rank_seconds")value=p[2]}
      if(name==wanted)found=value
    }
    END{if(found=="")exit 1;print found+0}' "${file}"
}

rss_field() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    $1=="PROCESS_RSS" && $2~/^label=sweep-/ {
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]==wanted)found=p[2]}}
    END{if(found=="")exit 1;print found+0}' "${file}"
}

printf 'case blas_threads wall_s initial_s sweep_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib\n' \
  >"${run_root}/performance.tsv"
for spec in 'blas1_a 1' 'blas4 4' 'blas8 8' 'blas16 16' 'blas1_b 1'; do
  set -- ${spec}; label=$1; threads=$2; file=${run_root}/${label}.out
  printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
    "${label}" "${threads}" "$(cat "${run_root}/${label}.case_wall_seconds")" \
    "$(segment_seconds "${file}" '^label=initial-full-chain$' initial_total)" \
    "$(segment_seconds "${file}" '^label=sweep-' sweep_total)" \
    "$(segment_seconds "${file}" '^label=sweep-' hamiltonian_action)" \
    "$(segment_seconds "${file}" '^label=sweep-' krylov)" \
    "$(segment_seconds "${file}" '^label=sweep-' svd)" \
    "$(segment_seconds "${file}" '^label=sweep-' block_update)" \
    "$(segment_seconds "${file}" '^label=sweep-' checkpoint_prepare)" \
    "$(segment_seconds "${file}" '^label=sweep-' checkpoint_finalize)" \
    "$(rss_field "${file}" sum_rank_high_water)" \
    "$(rss_field "${file}" max_rank_high_water)" \
    "$(awk 'NR>1&&$1~/^[0-9]+$/&&$2>p{p=$2}END{print p+0}' "${run_root}/${label}.storage.tsv")" \
    "$(awk 'NR>1&&$1~/^[0-9]+$/&&$3>p{p=$3}END{print p+0}' "${run_root}/${label}.storage.tsv")" \
    >>"${run_root}/performance.tsv"
done

awk '
  NR==2||NR==6 {base_wall+=$3;base_sweep+=$5;base_h+=$6;base_n++}
  NR>1 {label[NR]=$1;wall[NR]=$3;sweep[NR]=$5;h[NR]=$6}
  END {
    bw=base_wall/base_n;bs=base_sweep/base_n;bh=base_h/base_n
    print "case wall_speedup sweep_speedup h_action_speedup"
    for(i=2;i<=NR;i++)printf "%s %.9g %.9g %.9g\n",label[i],bw/wall[i],bs/sweep[i],bh/h[i]
  }' "${run_root}/performance.tsv" >"${run_root}/speedup.tsv"

find "${source_generation}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_after.tsv"
cmp "${run_root}/source_inventory_before.tsv" "${run_root}/source_inventory_after.tsv"

printf 'numerical=PASS\n' >"${run_root}/NUMERICAL_PASS"
printf 'benchmark=COMPLETE\n' >"${run_root}/BENCHMARK_COMPLETE"
printf 'hybrid_blas_d1000_scaling=PASS\n' >"${run_root}/PASS"
