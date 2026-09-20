#!/usr/bin/env bash
#SBATCH --job-name=tk_d5k_blas32
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=234G
#SBATCH --time=03:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_blas_d5000_bounded/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_blas_d5000_bounded/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root=${migration_root}/engine_leftedge_hybrid_bounded_v1_20260920
readonly converter_root=${migration_root}/converter_anchor_v2_20260918
readonly source_root=${migration_root}/leftedge_D5000_migration_20260917/half/itensor_converted
readonly d16_pass=${migration_root}/hybrid_bounded_d16_gate/run_12178658/PASS
readonly affinity_pass=${migration_root}/hybrid_thread_affinity_smoke/run_12175622/PASS
readonly test_root=${migration_root}/hybrid_blas_d5000_bounded
readonly batch_script=${test_root}/run_hybrid_blas_d5000_bounded.slurm
readonly direction_driver=${test_root}/bounded_direction_snapshot.jl
readonly run_root=${test_root}/run_${SLURM_JOB_ID}
readonly shm_parent=/dev/shm/${USER}/dmrg_d5k_blas32_${SLURM_JOB_ID}
readonly tmp_parent=/tmp/${USER}/dmrg_d5k_blas32_${SLURM_JOB_ID}
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${tmp_parent}"
  fi
}
trap cleanup EXIT INT TERM

test "$(cat "${d16_pass}")" = 'hybrid_bounded_d16_gate=PASS'
test "$(cat "${affinity_pass}")" = 'hybrid_thread_affinity_smoke=PASS'
test -f "${source_root}/1.jld2" && test -f "${source_root}/192.jld2"
test "$(find "${source_root}" -maxdepth 1 -type f | wc -l | awk '{print $1}')" -eq 193
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${tmp_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${tmp_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH=${migration_root}/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 TENSORKIT_TRANSFORMER_THREADS=1
export OMP_PLACES=cores OMP_PROC_BIND=close MKL_DYNAMIC=false
export model=${physical_label}
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=5000 INPUT_BOND_DIMENSION=5000 STOP_BOND_DIMENSION=5000
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR=${source_root}
export BENCHMARK_MAX_BONDS=32
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=32 ENV_PRIMARY_CAP_GB=64
export ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0 ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=true ENV_WINDOW_TRANSIENT_GB=0
export ENV_TIERED_PINGPONG=false
export HOPPING_CHANNEL_MODE=split_pruned H_ACTION_PIN_CACHE_GB=0
export H_ACTION_SCHEDULER=static_all_ranks MPI_TENSORMAP_REDUCE_MODE=packed
export ENGINE_ROOT=${engine_root}

sha256sum "${engine_root}/MPS/TN.jl" "${engine_root}/MPS.jl" \
  "${engine_root}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl" \
  "${batch_script}" "${direction_driver}" >"${run_root}/sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${run_root}/converter_sha256.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_before.tsv"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"
test "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" -ge 83886080
test "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" -ge 83886080

monitor_resources() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4
  printf 'unix_time primary_kib spill_kib combined_kib shm_available_kib tmp_available_kib step_maxrss_raw\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local p=0 s=0 maxrss=NA
    test ! -e "${primary}" || p=$(du -sk "${primary}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${spill}" || s=$(du -sk "${spill}" 2>/dev/null | awk '{print $1+0}')
    maxrss=$(sstat -j "${SLURM_JOB_ID}" --noheader --format=MaxRSS 2>/dev/null | awk 'NF{print $1;exit}')
    test -n "${maxrss}" || maxrss=NA
    printf '%s %s %s %s %s %s %s\n' "$(date +%s)" "${p}" "${s}" "$((p+s))" \
      "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" \
      "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" "${maxrss}" >>"${destination}"
    sleep 15
  done
}

run_case() {
  local label=$1 threads=$2
  local output=${run_root}/${label}_mps
  local primary=${shm_parent}/${label}
  local spill=${tmp_parent}/${label}_spill
  local work=${run_root}/${label}_work
  export OMP_THREADS=${threads} OMP_NUM_THREADS=${threads}
  export MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}
  export OUTPUT_MPS_DIR=${output}
  export ENVIRONMENT_TMP_ROOT=${primary}
  export ENVIRONMENT_SPILL_ROOT=${spill}
  export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_${label}
  export RUN_TAG=d5k_blas32_${label}_${SLURM_JOB_ID}
  mkdir -p "${work}/out/${model}" "${work}/out_short/${model}" \
    "${work}/time/${model}" "${work}/EE/${model}"
  {
    printf 'case=%s\nblas_threads=%s\nranks=4\ncpus_per_rank=16\n' "${label}" "${threads}"
    printf 'D=5000\nbenchmark_max_bonds=32\nsource=%s\noutput=%s\n' \
      "${source_root}" "${output}"
  } >"${run_root}/${label}.launch.txt"

  local started pid monitor_pid status=0
  started=$(date +%s)
  cd "${work}"
  srun --exclusive --mpi=pmi2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores \
    julia --project="${engine_root}" "${direction_driver}" \
    >"${run_root}/${label}.out" 2>"${run_root}/${label}.err" &
  pid=$!
  monitor_resources "${primary}" "${spill}" "${pid}" \
    "${run_root}/${label}.resources.tsv" &
  monitor_pid=$!
  wait "${pid}" || status=$?
  wait "${monitor_pid}" || true
  printf '%s\n' "$(( $(date +%s) - started ))" >"${run_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0

  grep -q "THREAD_CONFIGURATION julia=1 tensor_transformer=1 blas=${threads}" \
    "${run_root}/${label}.out"
  grep -q 'BOUNDED_SWEEP direction=forward max_bonds=32 total_bonds=191' \
    "${run_root}/${label}.out"
  grep -q 'BOUNDED_SWEEP_COMPLETE direction=forward processed_bonds=32 total_bonds=191' \
    "${run_root}/${label}.out"
  grep -q 'bounded_direction_snapshot=PASS direction=forward bonds=32' \
    "${run_root}/${label}.out"
  test "$(wc -l <"${work}/time/${model}"/*.txt | awk '{print $1}')" -eq 33
  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  current=${output}/$(tr -d '\n' <"${output}/CURRENT")
  test "$(find "${current}" -maxdepth 1 -type f | wc -l | awk '{print $1}')" -eq 193
  grep -q 'complete = true' "${current}/checkpoint_manifest.toml"
  grep -q 'partial_direction = true' "${current}/checkpoint_manifest.toml"
  grep -q 'processed_bonds = 32' "${current}/checkpoint_manifest.toml"
  cp "${current}/checkpoint_manifest.toml" "${run_root}/${label}.checkpoint_manifest.toml"
  du -sk "${output}" >"${run_root}/${label}.checkpoint_size_kib.txt"
  grep '^BOUNDED_DIRECTION_RESULT direction=forward ' "${run_root}/${label}.out" \
    >"${run_root}/${label}.result.txt"
  ! grep -Eq 'SpaceMismatch|BACKING_ERROR|OutOfMemoryError|Tensor file not found' \
    "${run_root}/${label}.out" "${run_root}/${label}.err"
}

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

cleanup_partial_mps() {
  local label=$1 target=${run_root}/$1_mps
  case "${target}" in
    "${run_root}"/*_mps) rm -rf -- "${target}" ;;
    *) echo "Refusing unsafe benchmark cleanup: ${target}" >&2; exit 2 ;;
  esac
  test ! -e "${target}"
}

compare_and_cleanup_candidate() {
  local candidate=$1
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
  cleanup_partial_mps "${candidate}"
}

# The two one-thread cases bracket cache/node drift; BLAS 4 is the D1000 wall
# winner and BLAS 16 is the D1000 H-action winner.  Retain only the first
# baseline while each candidate is compared, keeping peak project usage to two
# partial checkpoints rather than four.
run_case blas1_a 1
run_case blas4 4
compare_and_cleanup_candidate blas4
run_case blas16 16
compare_and_cleanup_candidate blas16
run_case blas1_b 1
compare_and_cleanup_candidate blas1_b

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

printf 'case blas_threads wall_s initial_s bounded_sweep_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib checkpoint_kib\n' \
  >"${run_root}/performance.tsv"
for spec in 'blas1_a 1' 'blas4 4' 'blas16 16' 'blas1_b 1'; do
  set -- ${spec}; label=$1; threads=$2; file=${run_root}/${label}.out
  printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
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
    "$(awk 'NR>1&&$1~/^[0-9]+$/&&$2>p{p=$2}END{print p+0}' "${run_root}/${label}.resources.tsv")" \
    "$(awk 'NR>1&&$1~/^[0-9]+$/&&$3>p{p=$3}END{print p+0}' "${run_root}/${label}.resources.tsv")" \
    "$(awk '{print $1+0}' "${run_root}/${label}.checkpoint_size_kib.txt")" \
    >>"${run_root}/performance.tsv"
done

awk '
  NR==2||NR==5 {bw+=$3;bi+=$4;bs+=$5;bh+=$6;bk+=$7;bv+=$8;n++}
  NR>1 {label[NR]=$1;wall[NR]=$3;initial[NR]=$4;sweep[NR]=$5;
    h[NR]=$6;k[NR]=$7;svd[NR]=$8}
  END {
    print "case wall_speedup initial_speedup bounded_sweep_speedup h_action_speedup krylov_speedup svd_speedup"
    for(i=2;i<=NR;i++)printf "%s %.9g %.9g %.9g %.9g %.9g %.9g\n",
      label[i],(bw/n)/wall[i],(bi/n)/initial[i],(bs/n)/sweep[i],
      (bh/n)/h[i],(bk/n)/k[i],(bv/n)/svd[i]
  }' "${run_root}/performance.tsv" >"${run_root}/speedup.tsv"

find "${source_root}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_after.tsv"
cmp "${run_root}/source_inventory_before.tsv" "${run_root}/source_inventory_after.tsv"

# These are explicitly disposable benchmark-only partial checkpoints.  All
# candidate outputs were released immediately after comparison; now release
# the retained baseline as well.
cleanup_partial_mps blas1_a
printf 'benchmark_partial_mps_cleanup=PASS\n' >"${run_root}/CLEANUP_PASS"

printf 'numerical=PASS\n' >"${run_root}/NUMERICAL_PASS"
printf 'benchmark=COMPLETE\n' >"${run_root}/BENCHMARK_COMPLETE"
printf 'hybrid_blas_d5000_bounded=PASS\n' >"${run_root}/PASS"
