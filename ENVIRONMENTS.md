# Julia environments

This directory is the TensorKit production-engine environment.  It is kept
separate from `converter/`, which is the compatibility bridge and therefore
also depends on ITensor.

## Production engine (BSCC x86_64)

Instantiate from this directory with Julia 1.11:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The production environment pins all direct dependencies.  The numerical core
is TensorKit 0.14.11, TensorOperations 5.6.1, MPI 0.20.24, and MKL 0.9.1;
FileIO/JLD2 and ProgressMeter are also locked.  `using MKL`
selects MKL through Julia's `libblastrampoline`; BLAS and LAPACK are not
separate Julia dependencies.

TensorKit 0.14.11 is intentionally different from the converter's 0.17.0.
It is the latest audited release that still exposes all legacy APIs used by
the supplied engine: positional-partition `tsvd`/`svd`, `leftorth`/
`rightorth`, and `TensorKit.MatrixAlgebra.svd!`.  TensorKit 0.15 removed the
last of those, while 0.17 also removed `tsvd`.  Upgrading the engine therefore
requires an explicit source port and new numerical regression tests.

Run the serial and two-rank environment checks with:

```sh
julia --project=. test/environment_smoke.jl
mpiexecjl --project=. -n 2 julia --project=. test/environment_smoke.jl
```

Run the canonical-basis and model-construction regressions with:

```sh
julia --project=. test/hubbard_u1u1_operators.jl
julia --project=. test/hubbard_u1u1_model_smoke.jl
julia --project=. test/checkpoint_lifecycle.jl
mpiexecjl --project=. -n 2 julia --project=. test/mpi_qn_block_reduce.jl
```

For the OBC left-edge adapter, set a unique `RUN_TAG`.  A converted MPS is
imported read-only with `START_MPS_DIR`; set a distinct `OUTPUT_MPS_DIR` for
the engine-owned result.  An existing output is rejected unless
`RESUME=true`, and resume also requires the recorded source identity to match.

Checkpoint sweeps use `generations/*.incoming` and publish only after full
readback and SHA-256 validation by atomically replacing `CURRENT`. `PREVIOUS`
remains available for recovery.  This correctness-first implementation copies
all site tensors into each incoming generation, so its I/O and temporary disk
cost must be measured before large-D production.

Numerical controls are explicit: `SVD_MODE=auto|serial|mpi` (or the compatible
boolean `SVD_MPI`), `KRYLOV_DIM`, `KRYLOV_TOL`, and `TRUNCATION_CUTOFF`.
`TRUNCATION_CUTOFF` is the squared discarded-norm target: TensorKit receives
`truncerr(sqrt(cutoff)) & truncdim(D)`, and the logged truncation error remains
the squared discarded singular-value norm.

Transient contraction environments are independent of MPS checkpoints.  For a
single-node Slurm job, use a job-private memory-first store and node-local
spill, for example:

```sh
export ENVIRONMENT_TMP_ROOT="/dev/shm/${USER}/dmrg_${SLURM_JOB_ID}"
export ENVIRONMENT_SPILL_ROOT="/tmp/${USER}/dmrg_spill_${SLURM_JOB_ID}"
export ENV_PRIMARY_RESERVE_GB=8
export ENV_LOCAL_CACHE_GB=0.125
export ENV_ROLLING_RELEASE=true
```

The primary store and spill must be distinct, non-nested, job-private paths.
`/public4` spill is rejected by default.  Node-local `/dev/shm` or `/tmp` is
also rejected when `SLURM_NNODES>1`, because dynamically scheduled MPI ranks
must share every environment key.  The bounded rank-local cache deep-copies
TensorMaps on insertion and hits so in-place contraction/reduction kernels do
not mutate cached state.  Only capacity errors (`ENOSPC`/`EDQUOT`) trigger
spill.  Consumed right/left environments are dropped after the complete
Krylov solve, and successful exit removes both backing roots after checking a
job-owner sentinel.  Keep `ENV_VERIFY_WRITES=true` for gates; production may
set it false only after the matching numerical gate passes.
The release schedule is validated only for the ground-state path and is
automatically disabled when `cal_excited=true`.

The smoke test checks the selected BLAS backend, a TensorOperations dense
contraction, a three-sector U(1) x U(1) TensorKit contraction and SVD,
JLD2 round-trip I/O, and MPI collective communication.  It does not validate
the Hubbard model or DMRG convergence.

The validated isolated BSCC copy is under
`/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/`.  Because the
login node could not resolve the Julia package servers during validation, use
the staged Linux depot explicitly:

```sh
export JULIA_DEPOT_PATH=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Set `JULIA_NUM_THREADS` and the MPI rank count explicitly for each benchmark;
do not infer useful parallelism from allocated CPUs.  Exact results and file
identities are in `test/VALIDATION_20260917.md`.

MKL is a production dependency for the BSCC x86_64 baseline.  Apple Silicon
development hosts should use the converter environment's OpenBLAS backend for
local representation tests rather than modifying the production lock.

## Converter bridge

The isolated bridge environment is under `converter/`.  Instantiate and test
it with:

```sh
cd converter
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. validate_roundtrip.jl SOURCE.h5 CONVERTED_DIR [Lx Ly U left_edge_h]
julia --project=. test_hubbard_operator_match.jl
```

Never point conversion output at an original checkpoint or a non-empty
directory.  The converter is currently a one-site-at-a-time dense prototype;
it is not approved for a D=20000 checkpoint.
