#!/usr/bin/env bash
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

: "${SOURCE_H5:?SOURCE_H5 is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${CASE_LABEL:?CASE_LABEL is required}"

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly converter_root="${migration_root}/converter"

test -f "${SOURCE_H5}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to overwrite existing OUTPUT_DIR=%s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
cd "${converter_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

printf 'case=%s\nsource=%s\noutput=%s\n' "${CASE_LABEL}" "${SOURCE_H5}" "${OUTPUT_DIR}"
/usr/bin/time -v julia --project=. itensor_to_tensorkit.jl "${SOURCE_H5}" "${OUTPUT_DIR}"
/usr/bin/time -v julia --project=. validate_roundtrip.jl \
  "${SOURCE_H5}" "${OUTPUT_DIR}" 32 6 12.0 0.1

