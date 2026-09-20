#!/usr/bin/env bash
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_rolling_20260918"
readonly converter_root="${migration_root}/converter"
readonly source_root="${migration_root}/small_mpi_regression/source_d16"
readonly gate_root="${migration_root}/rolling_environment_gate/run_${SLURM_JOB_ID}"
readonly shm_root="/dev/shm/${USER}/dmrg_rolling_${SLURM_JOB_ID}"
readonly spill_root="/tmp/${USER}/dmrg_rolling_spill_${SLURM_JOB_ID}"

cleanup() {
  rm -rf -- "${shm_root}" "${spill_root}" \
    "${gate_root}/tmp_legacy" "${gate_root}/tmp_rolling_disk"
}
trap cleanup EXIT

test -d "${source_root}"
test ! -e "${gate_root}"
mkdir -p "${gate_root}" "${shm_root}" "${spill_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${engine_root}"
df -h /dev/shm | tee "${gate_root}/dev_shm_df.txt"

export ENVIRONMENT_STORE_TEST_ROOT="${shm_root}/store_unit"
export ENVIRONMENT_SPILL_ROOT="${spill_root}/store_unit"
export ENV_PRIMARY_RESERVE_GB=1000
export ENV_STORE_TEST_FORCE_SPILL=true
export ENV_LOCAL_CACHE_GB=0.01 ENV_VERIFY_WRITES=true
julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. test/environment_store_mpi.jl`
    run(`$launcher -n 4 $child`)
' | tee "${gate_root}/environment_store_unit.txt"

export ENV_PRIMARY_RESERVE_GB=8
export ENV_STORE_TEST_FORCE_SPILL=false
export ENVIRONMENT_SPILL_ROOT="${spill_root}/dmrg"

export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14
export TRUNCATION_CUTOFF=1e-9 ENERGY_ONLY=false
export SWEEP_DIMS=16:16 FIRST_PASS_REVERSE=false
export START_MPS_DIR="${source_root}"
export ENV_VERIFY_WRITES=true

verify_program_cleanup() {
  local label=$1
  local output="${gate_root}/${label}.out"
  local primary spill
  primary=$(awk -F= '$1 == "ENVIRONMENT_STORE_ROOT" {print $2; exit}' "${output}")
  spill=$(awk -F= '$1 == "ENVIRONMENT_STORE_SPILL_ROOT" {print $2; exit}' "${output}")
  test -n "${primary}"
  test -n "${spill}"
  test ! -e "${primary}"
  test ! -e "${spill}"
  printf 'program_cleanup_%s=PASS primary=%s spill=%s\n' \
    "${label}" "${primary}" "${spill}" | tee "${gate_root}/${label}.cleanup.txt"
}

run_mpi_case() {
  local label=$1
  local output=$2
  local tmp_root=$3
  local rolling=$4
  local cache_gb=$5
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${tmp_root}"
  export ENV_ROLLING_RELEASE="${rolling}"
  export ENV_LOCAL_CACHE_GB="${cache_gb}"
  export RUN_TAG="${label}_${SLURM_JOB_ID}"
  export SVD_MODE=mpi
  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n 4 $child`)
  ' >"${gate_root}/${label}.out" 2>"${gate_root}/${label}.err"
  verify_program_cleanup "${label}"
}

run_serial_case() {
  local label=$1
  local output=$2
  local tmp_root=$3
  export OUTPUT_MPS_DIR="${output}"
  export ENVIRONMENT_TMP_ROOT="${tmp_root}"
  export ENV_ROLLING_RELEASE=true
  export ENV_LOCAL_CACHE_GB=0.25
  export RUN_TAG="${label}_${SLURM_JOB_ID}"
  export SVD_MODE=serial
  /usr/bin/time -v julia --project=. start.jl \
    >"${gate_root}/${label}.out" 2>"${gate_root}/${label}.err"
  verify_program_cleanup "${label}"
}

run_mpi_case legacy_mpi "${gate_root}/legacy_mpi" \
  "${gate_root}/tmp_legacy" false 0
run_mpi_case rolling_disk_mpi "${gate_root}/rolling_disk_mpi" \
  "${gate_root}/tmp_rolling_disk" true 0
run_mpi_case rolling_memory_mpi "${gate_root}/rolling_memory_mpi" \
  "${shm_root}/rolling_memory_mpi" true 0.25
run_serial_case rolling_memory_serial "${gate_root}/rolling_memory_serial" \
  "${shm_root}/rolling_memory_serial"

grep -Eq 'drops = 0' "${gate_root}/legacy_mpi.out"
grep -Eq 'drops = [1-9][0-9]*' "${gate_root}/rolling_disk_mpi.out"
grep -Eq 'drops = [1-9][0-9]*' "${gate_root}/rolling_memory_mpi.out"
grep -Eq 'drops = [1-9][0-9]*' "${gate_root}/rolling_memory_serial.out"

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' < "${checkpoint}/CURRENT")"
}

compare_pair() {
  local label=$1
  local bra=$2
  local ket=$3
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${bra}")" "$(current_generation "${ket}")" \
    | tee "${gate_root}/${label}.txt"
  local overlap fidelity
  local bra_norm ket_norm bra_dmax ket_dmax bra_blocks ket_blocks
  overlap=$(awk -F= '$1 == "normalized_overlap" {print $2}' "${gate_root}/${label}.txt")
  fidelity=$(awk -F= '$1 == "fidelity_per_site" {print $2}' "${gate_root}/${label}.txt")
  bra_norm=$(awk -F= '$1 == "bra_norm2" {print $2}' "${gate_root}/${label}.txt")
  ket_norm=$(awk -F= '$1 == "ket_norm2" {print $2}' "${gate_root}/${label}.txt")
  bra_dmax=$(awk -F= '$1 == "bra_max_bond_dimension" {print $2}' "${gate_root}/${label}.txt")
  ket_dmax=$(awk -F= '$1 == "ket_max_bond_dimension" {print $2}' "${gate_root}/${label}.txt")
  bra_blocks=$(awk -F= '$1 == "bra_max_block_count" {print $2}' "${gate_root}/${label}.txt")
  ket_blocks=$(awk -F= '$1 == "ket_max_block_count" {print $2}' "${gate_root}/${label}.txt")
  awk -v value="${overlap}" 'BEGIN { exit !(value >= 1.0 - 1.0e-11) }'
  awk -v value="${fidelity}" 'BEGIN { exit !(value >= 1.0 - 1.0e-12) }'
  awk -v bra="${bra_norm}" -v ket="${ket_norm}" 'BEGIN {
      difference=bra-ket; if (difference<0) difference=-difference
      exit !(difference <= 1.0e-10 && bra >= 1.0-1.0e-10 && bra <= 1.0+1.0e-10 &&
          ket >= 1.0-1.0e-10 && ket <= 1.0+1.0e-10)
  }'
  test "${bra_dmax}" = "${ket_dmax}"
  test "${bra_blocks}" = "${ket_blocks}"
  cd "${engine_root}"
}

compare_pair legacy_vs_rolling_disk \
  "${gate_root}/legacy_mpi" "${gate_root}/rolling_disk_mpi"
compare_pair legacy_vs_rolling_memory \
  "${gate_root}/legacy_mpi" "${gate_root}/rolling_memory_mpi"
compare_pair memory_serial_vs_mpi \
  "${gate_root}/rolling_memory_serial" "${gate_root}/rolling_memory_mpi"

printf 'rolling_environment_gate=PASS\n' | tee "${gate_root}/PASS"
