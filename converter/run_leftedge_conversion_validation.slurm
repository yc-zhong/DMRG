#!/usr/bin/env bash
#SBATCH --partition=amd_1T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=24:00:00

set -euo pipefail

: "${SOURCE_H5:?SOURCE_H5 is required}"
: "${CONVERTED_DIR:?CONVERTED_DIR is required}"
: "${CASE_LABEL:?CASE_LABEL is required}"

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly converter_root="${migration_root}/converter"

test -f "${SOURCE_H5}"
test -f "${CONVERTED_DIR}/conversion.toml"
test "$(find "${CONVERTED_DIR}" -maxdepth 1 -type f -name '*.jld2' | wc -l)" -eq 192
cd "${converter_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

printf 'stage=validate\ncase=%s\nsource=%s\nconverted=%s\n' \
  "${CASE_LABEL}" "${SOURCE_H5}" "${CONVERTED_DIR}"
/usr/bin/time -v julia --project=. validate_roundtrip.jl \
  "${SOURCE_H5}" "${CONVERTED_DIR}" 32 6 12.0 0.1

