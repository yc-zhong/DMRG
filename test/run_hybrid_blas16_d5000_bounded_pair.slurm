#!/usr/bin/env bash
#SBATCH --job-name=tk_d5k_blas16
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=234G
#SBATCH --time=03:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_blas16_d5000_bounded/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_blas16_d5000_bounded/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root=${migration_root}/engine_leftedge_hybrid_bounded_v1_20260920
readonly converter_root=${migration_root}/converter_anchor_v2_20260918
readonly source_root=${migration_root}/leftedge_D5000_migration_20260917/half/itensor_converted
readonly d16_run=${migration_root}/hybrid_bounded_d16_gate/run_12178658
readonly affinity_pass=${migration_root}/hybrid_thread_affinity_smoke/run_12175622/PASS
readonly test_root=${migration_root}/hybrid_blas16_d5000_bounded
readonly batch_script=${test_root}/run_hybrid_blas16_d5000_bounded_pair.slurm
readonly direction_driver=${test_root}/bounded_direction_snapshot.jl
readonly run_root=${test_root}/run_${SLURM_JOB_ID}
readonly shm_parent=/dev/shm/${USER}/dmrg_d5k_blas16_${SLURM_JOB_ID}
readonly tmp_parent=/tmp/${USER}/dmrg_d5k_blas16_${SLURM_JOB_ID}
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
readonly quota_reserve_kib=31457280

active_srun_pid=
active_monitor_pid=
cleanup_owned=false
benchmark_success=false

cleanup_node_local() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${tmp_parent}"
  fi
}

cleanup_partial_outputs() {
  local label target
  for label in blas1 blas16; do
    target=${run_root}/${label}_mps
    case "${target}" in
      "${run_root}"/*_mps) test ! -e "${target}" || rm -rf -- "${target}" ;;
      *) echo "Refusing unsafe benchmark cleanup: ${target}" >&2; return 2 ;;
    esac
  done
}

on_signal() {
  trap - INT TERM
  test -z "${active_srun_pid}" || kill -TERM "${active_srun_pid}" 2>/dev/null || true
  test -z "${active_srun_pid}" || wait "${active_srun_pid}" 2>/dev/null || true
  test -z "${active_monitor_pid}" || kill -TERM "${active_monitor_pid}" 2>/dev/null || true
  test -z "${active_monitor_pid}" || wait "${active_monitor_pid}" 2>/dev/null || true
  exit 143
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  if test "${benchmark_success}" != true; then
    cleanup_partial_outputs || true
    test ! -d "${run_root}" || printf 'failure_partial_mps_cleanup=PASS\n' \
      >"${run_root}/FAILURE_CLEANUP_PASS"
  fi
  cleanup_node_local
  exit "${status}"
}

trap on_signal INT TERM
trap on_exit EXIT

expect_sha256() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  test "${actual}" = "${expected}" || {
    echo "SHA256 mismatch for ${path}: ${actual} != ${expected}" >&2
    return 1
  }
}

check_quota_headroom() {
  local row used soft hard
  row=$(timeout 30s lfs quota -u "$(id -u)" /public4 | \
    awk '$1=="/public4"{print $2,$3,$4;exit}')
  test -n "${row}" || { echo 'Unable to read /public4 UID quota' >&2; return 1; }
  read -r used soft hard <<<"${row}"
  test "${soft}" -gt 0 && test "${hard}" -gt 0
  printf 'used_kib=%s soft_kib=%s hard_kib=%s required_headroom_kib=%s\n' \
    "${used}" "${soft}" "${hard}" "${quota_reserve_kib}"
  test $((used + quota_reserve_kib)) -le "${soft}" || {
    echo "Insufficient quota headroom: used=${used} soft=${soft}" >&2
    return 1
  }
}

test "$(cat "${d16_run}/PASS")" = 'hybrid_bounded_d16_gate=PASS'
test "$(cat "${affinity_pass}")" = 'hybrid_thread_affinity_smoke=PASS'
expect_sha256 b2168f40a6ea5f856efe36bc509bc5859d4ded290fc0403d32d1bcbcdbdfb7de \
  "${engine_root}/MPS.jl"
expect_sha256 44d5bff6750f0dde435732a527a1907774a7d78e86b5c156dab29c86d88b214a \
  "${engine_root}/MPS/TN.jl"
expect_sha256 dea1808450397a3af01ecb80ad33824735fffcb3be6ace46b5d1d3790eeac2bc \
  "${engine_root}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl"
expect_sha256 a7a4adad1a0cb0fdf97acbf41d78b1e43ad119f21147e0e78dcc384ad0542e0e \
  "${direction_driver}"
expect_sha256 f99b22deca5162b0dbb545d20b7447c786839fd37aad3eb60b0750e0db0ab0bf \
  "${source_root}/conversion.toml"
grep -q '^length = 192$' "${source_root}/conversion.toml"
grep -q '^total_Nf = 192$' "${source_root}/conversion.toml"
grep -q '^total_Sz_integer = 0$' "${source_root}/conversion.toml"
grep -q '^local_basis = \["Emp", "Up", "Dn", "UpDn"\]$' \
  "${source_root}/conversion.toml"
test -f "${source_root}/1.jld2" && test -f "${source_root}/192.jld2"
test "$(find "${source_root}" -maxdepth 1 -type f | wc -l | awk '{print $1}')" -eq 193
test ! -e "${run_root}" && test ! -e "${shm_parent}" && test ! -e "${tmp_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${tmp_parent}"
cleanup_owned=true
check_quota_headroom >"${run_root}/quota_before.txt"

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
cp "${source_root}/conversion.toml" "${run_root}/source_conversion.toml"
find "${source_root}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_before.tsv"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"
test "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" -ge 83886080
test "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" -ge 83886080

sample_kib() {
  local path=$1 value
  test -e "${path}" || { echo 0; return; }
  value=$(du -sk "${path}" 2>/dev/null | awk 'NF{print $1+0}' || true)
  echo "${value:-0}"
}

monitor_resources() {
  local primary=$1 spill=$2 watched_pid=$3 destination=$4 p s
  printf 'unix_time primary_sample_kib spill_sample_kib combined_sample_kib shm_available_kib tmp_available_kib\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    p=$(sample_kib "${primary}")
    s=$(sample_kib "${spill}")
    printf '%s %s %s %s %s %s\n' "$(date +%s)" "${p}" "${s}" "$((p+s))" \
      "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" \
      "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" >>"${destination}"
    sleep 10
  done
  printf 'monitor=PASS note=sampled_peaks_are_lower_bounds\n' >"${destination}.complete"
}

run_case() {
  local label=$1 threads=$2 output primary spill work started status=0 current
  output=${run_root}/${label}_mps
  primary=${shm_parent}/${label}
  spill=${tmp_parent}/${label}_spill
  work=${run_root}/${label}_work
  check_quota_headroom >"${run_root}/${label}.quota.txt"
  export OMP_THREADS=${threads} OMP_NUM_THREADS=${threads}
  export MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}
  export OUTPUT_MPS_DIR=${output}
  export ENVIRONMENT_TMP_ROOT=${primary}
  export ENVIRONMENT_SPILL_ROOT=${spill}
  export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_${label}
  export RUN_TAG=d5k_blas16_${label}_${SLURM_JOB_ID}
  mkdir -p "${work}/out/${model}" "${work}/out_short/${model}" \
    "${work}/time/${model}" "${work}/EE/${model}"
  printf 'case=%s\nblas_threads=%s\nranks=4\ncpus_per_rank=16\nD=5000\nbenchmark_max_bonds=32\nsource=%s\noutput=%s\n' \
    "${label}" "${threads}" "${source_root}" "${output}" \
    >"${run_root}/${label}.launch.txt"

  started=$(date +%s)
  cd "${work}"
  srun --exclusive --mpi=pmi2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores \
    julia --project="${engine_root}" "${direction_driver}" \
    >"${run_root}/${label}.out" 2>"${run_root}/${label}.err" &
  active_srun_pid=$!
  monitor_resources "${primary}" "${spill}" "${active_srun_pid}" \
    "${run_root}/${label}.resources.tsv" &
  active_monitor_pid=$!
  wait "${active_srun_pid}" || status=$?
  active_srun_pid=
  wait "${active_monitor_pid}"
  active_monitor_pid=
  printf '%s\n' "$(( $(date +%s) - started ))" \
    >"${run_root}/${label}.case_wall_seconds"
  test "${status}" -eq 0

  test -f "${run_root}/${label}.resources.tsv.complete"
  test "$(wc -l <"${run_root}/${label}.resources.tsv")" -ge 2
  test ! -e "${primary}" && test ! -e "${spill}"
  printf 'engine_environment_cleanup=PASS\n' >"${run_root}/${label}.cleanup.txt"
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
  cp "${current}/checkpoint_manifest.toml" \
    "${run_root}/${label}.checkpoint_manifest.toml"
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

numeric_value() {
  local key=$1 file=$2 value
  value=$(awk -F= -v key="${key}" '$1==key{print $2;found=1}END{if(!found)exit 1}' "${file}")
  [[ "${value}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || return 1
  awk -v x="${value}" 'BEGIN{exit !(x==x && x+0==x)}'
  printf '%s' "${value}"
}

result_field() {
  local key=$1 file=$2 value
  value=$(awk -v key="${key}" '{for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]==key)print p[2]}}' "${file}")
  [[ "${value}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || return 1
  awk -v x="${value}" 'BEGIN{exit !(x==x && x+0==x)}'
  printf '%s' "${value}"
}

run_case blas1 1
run_case blas16 16

cd "${converter_root}"
julia --project=. compare_tensorkit_mps.jl \
  "$(current_generation "${run_root}/blas1_mps")" \
  "$(current_generation "${run_root}/blas16_mps")" \
  >"${run_root}/blas1_vs_blas16.txt"
test "$(numeric_value length "${run_root}/blas1_vs_blas16.txt")" -eq 192
bra_norm=$(numeric_value bra_norm2 "${run_root}/blas1_vs_blas16.txt")
ket_norm=$(numeric_value ket_norm2 "${run_root}/blas1_vs_blas16.txt")
overlap=$(numeric_value normalized_overlap "${run_root}/blas1_vs_blas16.txt")
fidelity=$(numeric_value fidelity_per_site "${run_root}/blas1_vs_blas16.txt")
awk -v a="${bra_norm}" -v b="${ket_norm}" 'BEGIN{d=a-b;if(d<0)d=-d;s=(a>b?a:b);exit !(a>0&&b>0&&d<=1e-10*s)}'
awk -v x="${overlap}" 'BEGIN{exit !(x>=1-1e-10&&x<=1+1e-10)}'
awk -v x="${fidelity}" 'BEGIN{exit !(x>=1-1e-12&&x<=1+1e-12)}'
test "$(awk -F= '$1=="site_spaces_match"{print $2}' "${run_root}/blas1_vs_blas16.txt")" = true
test "$(numeric_value bra_max_bond_dimension "${run_root}/blas1_vs_blas16.txt")" = \
  "$(numeric_value ket_max_bond_dimension "${run_root}/blas1_vs_blas16.txt")"
test "$(numeric_value bra_max_block_count "${run_root}/blas1_vs_blas16.txt")" = \
  "$(numeric_value ket_max_block_count "${run_root}/blas1_vs_blas16.txt")"
energy1=$(result_field energy "${run_root}/blas1.result.txt")
energy16=$(result_field energy "${run_root}/blas16.result.txt")
trunc1=$(result_field truncation "${run_root}/blas1.result.txt")
trunc16=$(result_field truncation "${run_root}/blas16.result.txt")
awk -v a="${energy1}" -v b="${energy16}" 'BEGIN{d=a-b;if(d<0)d=-d;exit !(d<=1e-9)}'
awk -v a="${trunc1}" -v b="${trunc16}" 'BEGIN{d=a-b;if(d<0)d=-d;s=(a<0?-a:a);t=(b<0?-b:b);if(t>s)s=t;tol=1e-12;if(1e-8*s>tol)tol=1e-8*s;exit !(d<=tol)}'

segment_seconds() {
  local file=$1 pattern=$2 wanted=$3
  awk -v pattern="${pattern}" -v wanted="${wanted}" '
    $1=="SEGMENT_TIMING" && $2~pattern {
      name="";value=""
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]=="name")name=p[2];
        if(p[1]=="max_rank_seconds")value=p[2]}
      if(name==wanted)found=value
    }
    END{if(found==""||found!=found)exit 1;print found+0}' "${file}"
}

rss_field() {
  local file=$1 wanted=$2
  awk -v wanted="${wanted}" '
    $1=="PROCESS_RSS" && $2~/^label=sweep-/ {
      for(i=1;i<=NF;i++){split($i,p,"=");if(p[1]==wanted)found=p[2]}}
    END{if(found==""||found!=found)exit 1;print found+0}' "${file}"
}

printf 'case blas_threads wall_s initial_s bounded_sweep_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_sample_peak_kib spill_sample_peak_kib checkpoint_kib\n' \
  >"${run_root}/performance.tsv"
for spec in 'blas1 1' 'blas16 16'; do
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

awk 'NR==1{print "case wall_speedup initial_speedup bounded_sweep_speedup h_action_speedup krylov_speedup svd_speedup";next}
  NR==2{bw=$3;bi=$4;bs=$5;bh=$6;bk=$7;bv=$8;next}
  NR==3{printf "%s %.9g %.9g %.9g %.9g %.9g %.9g\n",$1,bw/$3,bi/$4,bs/$5,bh/$6,bk/$7,bv/$8}' \
  "${run_root}/performance.tsv" >"${run_root}/speedup.tsv"

find "${source_root}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_after.tsv"
cmp "${run_root}/source_inventory_before.tsv" "${run_root}/source_inventory_after.tsv"
cleanup_partial_outputs
test ! -e "${run_root}/blas1_mps" && test ! -e "${run_root}/blas16_mps"
printf 'benchmark_partial_mps_cleanup=PASS\n' >"${run_root}/CLEANUP_PASS"
printf 'numerical=PASS\n' >"${run_root}/NUMERICAL_PASS"
printf 'benchmark=COMPLETE\n' >"${run_root}/BENCHMARK_COMPLETE"
printf 'hybrid_blas16_d5000_bounded_equivalence=PASS\n' >"${run_root}/PASS"
benchmark_success=true
