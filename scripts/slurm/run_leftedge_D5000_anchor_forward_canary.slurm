#!/usr/bin/env bash
#SBATCH --job-name=tk_D5k_half_anchor
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=234G
#SBATCH --time=48:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/leftedge_D5000_anchor_20260918/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/leftedge_D5000_anchor_20260918/logs/%x.%j.err

set -euo pipefail

: "${D1000_ACCEPTANCE_PASS:?D1000_ACCEPTANCE_PASS is required}"

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v2_20260918"
readonly source_root="${migration_root}/leftedge_D5000_migration_20260917/half/itensor_converted"
readonly run_parent="${migration_root}/leftedge_D5000_anchor_20260918/half"
readonly run_root="${run_parent}/forward_r4_job_${SLURM_JOB_ID}"
readonly output_mps="${run_root}/mps"
readonly shm_parent="/dev/shm/${USER}/dmrg_D5000_anchor_half_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_D5000_anchor_half_${SLURM_JOB_ID}"
readonly primary="${shm_parent}/environment"
readonly spill="${local_parent}/environment_spill"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -f "${D1000_ACCEPTANCE_PASS}"
grep -qx 'd1000_anchor_acceptance=PASS' "${D1000_ACCEPTANCE_PASS}"
readonly acceptance_dir="$(dirname "${D1000_ACCEPTANCE_PASS}")"
test -f "${acceptance_dir}/engine_core_sha256.txt"
test -f "${source_root}/conversion.toml"
test "$(find "${source_root}" -maxdepth 1 -type f -name '[0-9]*.jld2' | wc -l)" -eq 192
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

cd "${engine_root}"
sha256sum -c "${acceptance_dir}/engine_core_sha256.txt" \
  | tee "${run_root}/engine_hash_check.txt"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl start.jl \
  >"${run_root}/engine_core_sha256.txt"
sha256sum "${source_root}/conversion.toml" >"${run_root}/source_conversion_sha256.txt"
du -sk "${source_root}" >"${run_root}/source_size_kib.txt"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model="${physical_label}"
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=5000 INPUT_BOND_DIMENSION=5000 STOP_BOND_DIMENSION=5000
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=5000 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export OUTPUT_MPS_DIR="${output_mps}"
export ENVIRONMENT_TMP_ROOT="${primary}"
export ENVIRONMENT_SPILL_ROOT="${spill}"
export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_half_D5000_anchor_forward"
export ENV_PRIMARY_RESERVE_GB=16
export ENV_ROLLING_RELEASE=true
export ENV_LOCAL_CACHE_GB=0.125
export ENV_VERIFY_WRITES=false
export ENVIRONMENT_WINDOW_SITES=32
export ENVIRONMENT_WINDOW_ANCHORS=true
export PROFILE_SEGMENTS=true
export RUN_TAG="leftedge_half_D5000_anchor_forward_${SLURM_JOB_ID}"

df -h /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"
shm_available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4+0}')
tmp_available_kib=$(df -Pk /tmp | awk 'NR==2 {print $4+0}')
persistent_available_kib=$(df -Pk "${migration_root}" | awk 'NR==2 {print $4+0}')
read -r quota_used_kib quota_limit_kib < <(lfs quota -u "${USER}" /public4 | awk '$1=="/public4" {print $2,$4;exit}')
test -n "${quota_used_kib}" && test -n "${quota_limit_kib}"
quota_free_kib=$((quota_limit_kib - quota_used_kib))
test "${shm_available_kib}" -ge 83886080
test "${tmp_available_kib}" -ge 83886080
test "${persistent_available_kib}" -ge 41943040
test "${quota_free_kib}" -ge 41943040

{
  printf 'job_id=%s\ncase=half\ngeometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n' "${SLURM_JOB_ID}"
  printf 'U=12.0\nt=1.0\nty=1.0\npinning=left_edge_af\npinning_strength=0.1\n'
  printf 'Ne=192\nSz2=0\nD=5000\nsweep_dims=5000\nfirst_pass_reverse=false\n'
  printf 'purpose=one_forward_pass_resource_canary_not_full_bidirectional_sweep\n'
  printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff=1e-9\nsvd_mode=mpi\n'
  printf 'ranks=4\nthreads_per_rank=1\nslurm_cpus=64\nslurm_mem=234G\nslurm_time=48:00:00\n'
  printf 'environment_window_sites=32\nenvironment_window_anchors=true\n'
  printf 'start_mps=%s\noutput_mps=%s\n' "${source_root}" "${output_mps}"
  printf 'd1000_acceptance_pass=%s\nprimary=%s\nspill=%s\n' \
    "${D1000_ACCEPTANCE_PASS}" "${primary}" "${spill}"
  printf 'primary_reserve_gb=16\ncache_gb_per_rank=0.125\nrolling_release=true\n'
  printf 'shm_available_kib=%s\ntmp_available_kib=%s\npersistent_available_kib=%s\n' \
    "${shm_available_kib}" "${tmp_available_kib}" "${persistent_available_kib}"
  printf 'quota_used_kib=%s\nquota_limit_kib=%s\nquota_free_kib=%s\n' \
    "${quota_used_kib}" "${quota_limit_kib}" "${quota_free_kib}"
  printf 'memory_status=unvalidated_D5000_capacity_bounded_by_slurm_234G_resource_canary\n'
} | tee "${run_root}/launch_manifest.txt"

monitor_storage() {
  local watched_pid=$1 destination=$2
  local valid_samples=0 failed_samples=0
  printf 'unix_time primary_kib spill_kib shm_used_kib shm_available_kib tmp_used_kib tmp_available_kib\n' \
    >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0 du_output
    if test -e "${primary}"; then
      if ! du_output=$(du -sk "${primary}" 2>/dev/null); then
        failed_samples=$((failed_samples + 1)); sleep 30; continue
      fi
      primary_kib=$(awk '{print $1+0}' <<<"${du_output}")
    fi
    if test -e "${spill}"; then
      if ! du_output=$(du -sk "${spill}" 2>/dev/null); then
        failed_samples=$((failed_samples + 1)); sleep 30; continue
      fi
      spill_kib=$(awk '{print $1+0}' <<<"${du_output}")
    fi
    printf '%s %s %s ' "$(date +%s)" "${primary_kib}" "${spill_kib}" >>"${destination}"
    df -Pk /dev/shm | awk 'NR==2 {printf "%s %s ", $3, $4}' >>"${destination}"
    df -Pk /tmp | awk 'NR==2 {print $3, $4}' >>"${destination}"
    valid_samples=$((valid_samples + 1))
    sleep 30
  done
  printf '# monitor_summary valid_samples=%s failed_samples=%s\n' \
    "${valid_samples}" "${failed_samples}" >>"${destination}"
}

/usr/bin/time -v julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/engine.out" 2>"${run_root}/engine.err" &
run_pid=$!
monitor_storage "${run_pid}" "${run_root}/storage.txt" &
monitor_pid=$!
status=0 monitor_status=0
wait "${run_pid}" || status=$?
wait "${monitor_pid}" || monitor_status=$?
test "${status}" -eq 0
test "${monitor_status}" -eq 0

valid_samples=$(awk -F'[ =]' '/^# monitor_summary/ {for(i=1;i<=NF;i++)if($i=="valid_samples")print $(i+1)}' "${run_root}/storage.txt")
test -n "${valid_samples}" && test "${valid_samples}" -ge 10
test -f "${output_mps}/CURRENT" && test -f "${output_mps}/PREVIOUS"
generation="${output_mps}/$(tr -d '\n' <"${output_mps}/CURRENT")"
test -f "${generation}/checkpoint_manifest.toml"
grep -Eq '^complete = true$' "${generation}/checkpoint_manifest.toml"
test "$(find "${generation}" -maxdepth 1 -type f -name '[0-9]*.jld2' | wc -l)" -eq 192
grep -Eq '^reverse = false$' "${generation}/checkpoint_manifest.toml"
grep -Eq '^bond_dimension = 5000$' "${generation}/checkpoint_manifest.toml"
grep -Eq '^environment_window_sites = 32$' "${generation}/checkpoint_manifest.toml"
grep -Eq '^environment_window_anchors = true$' "${generation}/checkpoint_manifest.toml"
grep -Eq 'WINDOW_WORK direction=forward.*window=32 anchors=true.*prepare_calls=6.*environment_site_updates=343.*anchor_snapshots=5.*anchor_restores=5' "${run_root}/engine.out"
grep -q 'SEGMENT_TIMING.*name=sweep_total' "${run_root}/engine.out"
grep -q 'PROCESS_RSS.*sum_rank_high_water=' "${run_root}/engine.out"
grep -Eq 'drops = [1-9][0-9]*' "${run_root}/engine.out"
! grep -Eq 'missing_drops = [1-9][0-9]*' "${run_root}/engine.out"

program_primary=$(awk -F= '$1=="ENVIRONMENT_STORE_ROOT" {print $2;exit}' "${run_root}/engine.out")
program_spill=$(awk -F= '$1=="ENVIRONMENT_STORE_SPILL_ROOT" {print $2;exit}' "${run_root}/engine.out")
test -n "${program_primary}" && test -n "${program_spill}"
test ! -e "${program_primary}" && test ! -e "${program_spill}"

short_file="${engine_root}/out_short/${physical_label}/MPS_Lx32_Ly6_Ne192_Sz20_U12.0_t1.0_ty1.0_leftedge0.1_run${RUN_TAG}.txt"
test -s "${short_file}"
cp "${short_file}" "${run_root}/out_short.txt"
tr -d '[] ' <"${run_root}/out_short.txt" >"${run_root}/out_short.csv"
awk -F, '
  function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
  NR>1 {rows++; if(NF!=6)bad=1; for(i=1;i<=6;i++)if(!finite($i))bad=1; if($1+0!=1 || $2+0!=5000)bad=1}
  END {exit !(rows==1 && bad==0)}' "${run_root}/out_short.csv"
primary_peak=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ && $2>p {p=$2} END {print p+0}' "${run_root}/storage.txt")
spill_peak=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ && $3>p {p=$3} END {print p+0}' "${run_root}/storage.txt")
du -sk "${output_mps}" >"${run_root}/output_mps_size_kib.txt"

{
  printf 'forward_resource_canary=PASS\n'
  printf 'primary_peak_kib=%s\nspill_peak_kib=%s\n' "${primary_peak}" "${spill_peak}"
  printf 'output_mps=%s\n' "${output_mps}"
  printf 'note=checkpoint_saved_but_full_bidirectional_D5000_sweep_not_yet_complete\n'
} | tee "${run_root}/PASS"
