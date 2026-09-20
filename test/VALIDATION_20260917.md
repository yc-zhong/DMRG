# Environment validation record — 2026-09-17

The validation began with small synthetic tensors and a copied 32-site test
checkpoint.  Later explicitly authorized BSCC stages converted both existing
production D1000 checkpoints read-only and ran bounded MPI correctness and
Hamiltonian-evaluation jobs.  No production source checkpoint was modified or
deleted, and no D20000 job was submitted.

## Isolated BSCC deployment

The test installation is under
`/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/`:

- engine project: `engine/`
- converter project: `converter/`
- clean offline depot: `depot_final/`

BSCC could not resolve the Julia package servers or GitHub during the test, so
the locked projects, package sources, and platform artifacts were staged from
the local machine. The final Linux depot was rebuilt with AppleDouble files
excluded; `APPLEDOUBLE_COUNT=0` was verified. Earlier staging depots were left
untouched rather than deleted.

## Locked environments

Both projects use Julia 1.11.5.

| Environment | Direct numerical dependencies |
|---|---|
| Engine | TensorKit 0.14.11, TensorOperations 5.6.1, MPI 0.20.24, MKL 0.9.1, JLD2 0.6.4 |
| Converter | TensorKit 0.17.0, TensorOperations 5.6.2, ITensors 0.9.30, ITensorMPS 0.4.1, HDF5 0.17.3, JLD2 0.6.4 |

TensorKit 0.14.11 is intentional for the unported engine: it is the latest
audited release retaining positional `tsvd`/`svd`, `leftorth`/`rightorth`, and
`TensorKit.MatrixAlgebra.svd!`. The converter does not use those legacy APIs.

## Engine smoke tests on BSCC

Serial and two-rank tests passed with the clean offline depot. The two-rank run
reported:

- Julia 1.11.5; TensorKit 0.14.11; TensorOperations 5.6.1; MPI 0.20.24;
- MKL selected for both LP64 and ILP64 BLAS/LAPACK;
- MPICH 5.0.1, two ranks, rank all-reduce sum 3;
- dense TensorOperations contraction completed;
- three U(1) x U(1) sectors were retained;
- block-sparse contraction dense-reference error `2.719e-16`;
- TensorKit SVD reconstruction error `3.190e-16`;
- JLD2 round-trip completed.

The first invocation of the supplied `Square_Hubbard_U1_U1_CBC` example on a
2x2 case at `U=12`, `Ne=4` failed while constructing `Hopping1`: its dense
operators used `(Emp,UpDn,Up,Dn)`, while TensorKit canonically lays out the
declared sectors as `(Emp,Up,Dn,UpDn)`. This blocker was subsequently fixed as
described below. Passing the generic tensor smoke test alone still does not
make the supplied model production-ready.

## Canonical-basis repair

`model/Hubbard_U1_U1_operators.jl` now applies the exact permutation
`p=(1,3,4,2)` to both indices of every supplied local matrix. The TensorKit
physical space and converted MPS remain in `(Emp,Up,Dn,UpDn)` order. Existing
`Hopping1`/`Hopping2` signs and parity-string placement were retained.

The old TensorKit API calls in `MPS/TN.jl` and the model split helpers were
also updated from the unavailable three-result `svd` form to four-result
`tsvd`, explicitly ignoring only the returned discarded-weight diagnostic.
No cutoff or physical parameter was changed.

Validation results:

- local operator/QN/algebra/SVD tests: 46/46 passed;
- local ITensor Electron two-site element-by-element comparison: 10/10 passed,
  with maximum hopping-matrix difference exactly zero;
- the two-site hopping spectrum is
  `[-2,-1×4,0×6,1×4,2]` and is Hermitian;
- local 2x2 `U=12`, `Ne=4` model construction: 6/6 passed;
- `Hopping1` and `Hopping2` split/reconstruction errors were both
  `8.455206652451151e-16`;
- isolated BSCC engine operator tests: 46/46 passed;
- isolated BSCC 2x2 model construction: 6/6 passed with the same reconstruction
  errors.

The ITensor comparison also confirms that the supplied local `C_dn` is a
commuting-species hard-core operator; its doublon matrix element must remain
positive. The existing asymmetric parity factors recover ITensor's fermionic
hopping exactly. This repair closes the local-basis constructor failure, not
the missing OBCxOBC column-snake `left_edge_af` production adapter.

An initial relative-path rsync placed a redundant copy under the isolated
`engine/DMRG/` staging tree before the files were copied to their correct
`engine/{MPS,model,measure,test}` paths.  After explicit user authorization,
the 100 KiB redundant subtree was deleted and verified absent.  The active
engine files and all checkpoints were untouched.

## Target OBCxOBC left-edge adapter and small-system benchmarks

The target implementation is split into
`model/Hubbard_OBC_LeftEdge_U1_U1_core.jl` and
`model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl`.  It uses the production
column-snake map, OBC in both directions, nearest-neighbor `t` and `ty`, onsite
`U*Nupdn`, and `h*(-1)^(x+y)*(Nup-Ndn)` only on `x=1`.  Its U1xU1 output charge
is `(Ne,Nup-Ndn)` and its hopping `tsvd` is untruncated.

Local validation commands:

```bash
julia +1.11.5 --project=DMRG DMRG/test/hubbard_u1u1_operators.jl
julia +1.11.5 --project=DMRG DMRG/test/hubbard_obc_adapter_ed.jl
julia +1.11.5 --project=DMRG/converter DMRG/test/hubbard_obc_itensor_ed.jl
julia +1.11.5 --project=DMRG/converter DMRG/test/hubbard_obc_sameD_itensor.jl
```

Results:

- canonical local operators: 55/55;
- adapter geometry, OBC bond set, pinning, exact U=0 reference, term tables,
  onsite matrices, production 32x6 sectors, and hopping reconstruction: 82/82;
- the same 82/82 adapter/ED suite passed in the isolated BSCC engine
  environment with one thread; no Slurm job was submitted;
- direct construction of both 192-site product MPS completed, with final
  virtual charges `(192,0)` for half filling and `(180,0)` for doping;
- 3x2 U=0 fixed `(Nup,Ndn)=(3,3)` ED space dimension 400; adapter/reference
  energy agreement `1.78e-15` for the test's `t=0.9,ty=0.7,h=0.1` point;
- 2x2 U=12, `t=1,ty=0.7,h=0.1`: independent ED versus ITensor maximum
  aligned matrix error `1.11e-16`, maximum full-spectrum error `3.91e-14`;
- actual new-engine 3x2 U=0, `t=ty=1,h=0.1,D=64`, three sweeps: total
  energy `-7.662581575825796`, exact `-7.6625815761783915`, error `3.53e-10`;
- actual new-engine 3x2 U=12, `D=16`, four directional passes: total energy
  `-1.607032399101363`, final reported maximum truncation indicator
  `1.1855e-3`.  One ITensor sweep contains both directions, so the
  matched-update comparison is two ITensor sweeps, which gave
  `-1.6070639562099738`; exact ED is `-1.622432126236072`.  The older
  four-pass versus four-ITensor-sweep result is retained only as an unmatched
  diagnostic;
- the U12/D16 output MPS has norm 1, total N=6 and total Sz=0.  Direct overlap
  with the matched-update ITensor final MPS is `0.9999924799167476`.  Maximum
  differences are `8.41e-6` for local density, `1.7432e-3` for local Sz,
  `8.08e-5` for local double occupancy, `6.43e-5` for selected connected
  charge correlators, and `8.41e-4` for the corresponding physical-Sz
  connected correlators.

New-engine output is under
`TNc/Square_Hubbard_U1_U1_OBC_LeftEdge/` and
`out_short/Square_Hubbard_U1_U1_OBC_LeftEdge/`.  These are disposable small
test states, not converted or production checkpoints.  The first-sweep/JIT
times dominate this tiny case, so no performance conclusion is drawn.  No
Slurm job was submitted.

The earlier first-block-only reductions in `MPS.jl` have been replaced by a
common collective that verifies block count/order/size and reduces every QN
block through contiguous MPI buffers.  `test/mpi_qn_block_reduce.jl` verifies
all four physical U(1)xU(1) blocks with two local MPI ranks. This validates the
collective itself, not yet a complete multi-rank DMRG sweep or scaling claim.

Checkpoint I/O now uses immutable generations: a sweep constructs and checks
`generations/*.incoming`, writes a SHA-256 manifest, then atomically publishes
the `CURRENT` pointer while retaining `PREVIOUS`. The local lifecycle test
covers read-only import, an interrupted unpublished generation, checksum
detection, recovery, physical-leg and virtual-bond validation. Full copying is
currently correctness-first and requires cluster I/O measurement before large
bond dimension use.

A local `start.jl` regression completed two serial 3x2, U=12, D=16 sweeps
with `KRYLOV_DIM=3`, `KRYLOV_TOL=1e-12`, and zero cutoff. The published
`CURRENT` is `sweep-2-D16`, `PREVIOUS` is `sweep-1-D16`, all six tensors pass
the manifest readback, and the per-site energies changed from
`-0.19330993869363186` to `-0.22099487521384306`. This is a checkpoint/control
flow regression only, not a convergence or speed result.

## Converter round trip on BSCC

Input was a copy of the local 32-site half-filled test checkpoint:

`converter/smoke_input/mps_Lx16_Ly2_U6.000_Ne32_L16_Ly2_half_U6_m2000_last.h5`

Output was written only to `converter/smoke_output_half/`. Results were:

- length 32, `Nf=32`, integer twice-spin `0`;
- source/restored norm `1.0000000000000029`;
- TensorKit norm `1.0000000000000022 + 2.94e-17im`;
- normalized overlap `1` and maximum round-trip tensor difference `0`;
- maximum local-density difference `1.554e-15`;
- maximum local-spin difference `4.113e-16`;
- maximum double-occupancy difference `2.359e-16`;
- maximum selected charge-correlator difference `1.346e-15`;
- maximum selected spin-correlator difference `9.012e-19`;
- source/restored energy `-9.864427305145913`, difference `0`.

This establishes the small-state representation conversion and cluster
readback. It does not establish memory feasibility at D=5000 or D=20000, nor
that a converted production MPS can be swept safely through the target
OBCxOBC `left_edge_af` model adapter and restart path.

## Production D1000 lossless conversion on BSCC

The already existing ITensor D1000 stages were reverified before conversion.
Both are 32x6, OBCxOBC, column-snake, U=12, `left_edge_af,h=0.1`, and integer
twice-spin zero.  The source files remain unchanged under
`hubbard_ladder_benchmark_output/mps_checkpoints/leftedgeaf_h010_L32_Ly6_D4000/`.
Their original D1000 stages completed six sweeps but were not converged: the
last energy decreases were about 3.56 (half) and 3.80 (doped).

Read-only conversion/round-trip jobs wrote only under
`new_engine_migration/d1000_validation/`:

| case | old ITensor job | conversion job | source energy | overlap | energy difference | wall | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| half, Ne=192 | 12144833 | 12162192 | -54.52885574604858 | 1.0000000000000002 | 2.84e-14 | 00:08:38 | 3,799,412 KiB |
| doped, Ne=180 | 12144832 | 12162193 | -83.33771616203789 | 1.0000000000000004 | 4.26e-14 | 00:08:33 | 3,737,244 KiB |

Both conversions had exact per-tensor round trips.  TensorKit-native norms
agreed with the source at about 1e-14; total N was 192/180, total physical Sz
was zero at roundoff, and maximum tested local-density, local-Sz, doublon and
selected connected charge/spin discrepancies were at most 5.0e-15.  Slurm
allocated a whole 64-CPU node to each job, but the scripts intentionally used
one Julia/BLAS thread; TotalCPU was only 7:37 and 7:31.  These numbers are
conversion correctness and resource measurements, not a 64-core speed result.

During the first multi-worker engine launch, audit found that the legacy
Hamiltonian and SVD dispatchers reused one asynchronous one-element send
buffer across workers.  The affected D1000 jobs 12162242 and 12162243 were
cancelled after 54 seconds before accepting any result.  Dispatch now uses a
distinct blocking buffer per worker.  The MPI SVD path also required owned
dense matrices for TensorKit block views, matrix-valued SectorDict entries,
and TensorKit 0.14 `_compute_truncdim`/`_compute_truncerr` rather than the
nonexistent `_truncate!`.

The full-pass gate subsequently passed in Slurm job 12162354.  Both paths
started from the identical six-site D16 checkpoint and used U=12,
`left_edge_af,h=0.1`, Krylov dimension 3 and squared discarded-norm cutoff
1e-9.  Serial and 8-rank results were:

- serial `E/site=-0.2408852512692918`, reported pass time 43.91 s;
- 8-rank `E/site=-0.24088525126929175`, reported pass time 86.29 s;
- both norms `1.0000000000000013`;
- normalized overlap and per-site fidelity exactly 1 at printed precision;
- both maximum bond dimension 16 and maximum QN-block count 9.

Thus the complete multi-worker Hamiltonian, all-block reduction and MPI SVD
path reproduces the serial state on this gate.  It is deliberately not a
speed claim: at D16 MPI overhead dominates and the 8-rank pass is about twice
as slow.  Although the amd_test node was accounted as 64 allocated CPUs, the
launcher used 8 MPI ranks and one Julia/BLAS thread per rank; the Slurm step
reported TotalCPU 28:46, wall 4:02 including startup/precompilation, and peak
RSS 7,263,292 KiB across the step.

## Production D1000 new-engine Hamiltonian evaluation on BSCC

After the full-pass gate passed, `ENERGY_ONLY=true` evaluated the exact
OBCxOBC, column-snake, U=12, `left_edge_af,h=0.1` Hamiltonian on each converted
D1000 state.  This exercised the new engine's distributed term contractions
and all-QN-block reductions without changing the MPS.  Each launcher used 8
MPI ranks and one Julia/BLAS thread per rank:

| case | job | fresh ITensor energy | new-engine energy | absolute difference | new norm squared | wall | step peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| half, Ne=192 | 12162365 | -54.52885574604858 | -54.52885574604852 | about 6e-14 | 0.9999999999999998 | 00:04:29 | 14,266,612 KiB |
| doped, Ne=180 | 12162366 | -83.33771616203789 | -83.33771616203741 | about 4.8e-13 | 0.9999999999999997 | 00:04:43 | 13,322,140 KiB |

The immutable imported generations each contain 192 site files and a complete
SHA-256 manifest; `CURRENT` points to those generations.  The source ITensor
files retain their original sizes and timestamps: 396,693,073 bytes at
2026-09-14 15:03:14 for half and 405,893,262 bytes at 2026-09-14 15:06:40 for
doped.  No source was overwritten or deleted.

Slurm accounted the whole node as 64 allocated CPUs, but these were 8-rank
jobs.  Step TotalCPU was 32:37.596 and 34:16.547, corresponding to about 91%
CPU efficiency relative to the 8 launched ranks, not relative to 64 allocated
CPUs.  Aggregate step I/O was about 50/16.6 GB read/write for half and
51.4/17.0 GB for doped.  This validates D1000 representation plus Hamiltonian
evaluation; it is not a fixed-D1000 DMRG continuation or a 64-rank scaling
test.  The pre-existing D1000 states remain unconverged, and no D20000 job was
submitted.

## Production D5000 conversion validation and blocked sweep

The latest compatible 32x6 `left_edge_af,h=0.1` D5000 sources were converted
without modifying the ITensor HDF5 files.  Full `amd_256` round-trip jobs gave:

| case | validation job | normalized overlap | max tensor difference | N | Sz | energy difference | wall | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| half | 12164072 | 0.9999999999999998 | 0 | 192.0000000000006 | 9.99e-15 | 1.42e-14 | 01:44:40 | 33,048,388 KiB |
| doped | 12164073 | 1.0000000000000002 | 0 | 180.00000000000037 | 6.11e-16 | 0 | 01:46:07 | 38,638,520 KiB |

Maximum local-density, local-Sz, doublon, selected charge-correlator and
spin-correlator discrepancies were no larger than `5.33e-15`.  Independent
four-rank contiguous two-pass gate 12164071 also passed: both norms were one at
roundoff, normalized overlap and per-site fidelity were 1, and both Dmax and
maximum QN-block count matched.

Dependency-released D5000 jobs 12164191/12164192 are not completed sweeps.  Both
failed before the first variational update while constructing right
environments, with `SystemError: close: Disk quota exceeded`.  Peak MPI-step RSS
was only about 9.63 GB, excluding memory as the cause.  The identified half and
doped temporary trees are under the user-controlled `zyc` project, contain 826
and 617 files, and occupy 81,169,816 and 65,852,284 KiB.  The Lustre UID quota
reported 524,600,048 KiB used against a 524,288,000 KiB hard limit; this does
not imply ownership or control of the rest of `/public4`.  Their `CURRENT`
pointers reference complete 192-site imports of the verified starting states,
not new DMRG checkpoints.  No energy lowering, truncation error or D5000 engine
speed comparison can be claimed from these failed jobs.

With explicit authorization on 2026-09-18, the two failed temporary trees and
the two failed imported-output directories were removed by exact path and
verified absent.  UID usage decreased from 524,600,048 to 364,042,528 KiB,
releasing 160,557,520 KiB (about 153 GiB).  The verified 6.2/6.9 GiB converted
half/doped starts and all original ITensor checkpoints were retained.

Rolling-environment gate 12165661 established one narrow positive result:
with the cache disabled, immediate post-Krylov release produced a final D16
MPS identical to the legacy retention path (normalized overlap and per-site
fidelity 1).  The same job rejected the initial cached path, whose normalized
overlap with legacy was only 0.1558633; therefore 12165661 is an overall FAIL
and gives no authorization for D5000.  Investigation found that non-TensorMap
Krylov helper objects and mutable TensorMap storage could be returned stale or
aliased from a rank-local cache.  Replacement gate 12165695 uses TensorMap-only
cache admission, deep-copy isolation, forced `/dev/shm` to `/tmp` spill tests,
cross-rank generic `Ham_save` readback, exact duplicate detection, and
job-owner cleanup sentinels.  Gate 12165695 was subsequently cancelled after review
identified another stale-cache window: rank 0 overwrote the same Lanczos-vector
key each Krylov iteration without invalidating non-root caches.  Corrected gate
12165765 adds an all-rank invalidate/barrier after every such overwrite,
exclusive sentinel creation, unsafe-root rejection, positive drop-count and
post-cleanup assertions, norm equality, and exact Dmax/QN-block checks.  Its
legacy calculation passed, but an over-broad shell assertion rejected the
removed environment parent; 12165789 then exposed a pre-`MPI.Init` same-owner
sentinel race and was cancelled before rolling work.  Both failures were fixed
without weakening the intended assertions.

Final gate 12165817 COMPLETED in 13:56 and wrote
`rolling_environment_gate=PASS`.  Its forced-spill/cross-rank unit passed.
Legacy MPI, rolling-disk MPI and rolling-memory MPI wall times were 2:57.45,
2:55.72 and 2:55.08; rolling-memory serial was 2:26.50.  Rolling MPI owner
ranks reported positive drop counts (10, 18, 21 and 25 across the distributed
owners; serial reported 74), while legacy reported zero.  Every exact primary
and spill root was absent immediately after its case.  Legacy versus
rolling-disk, legacy versus rolling-memory, and memory serial versus MPI all
gave normalized overlap/fidelity 1, norm squared one at roundoff, Dmax 16 and
maximum QN-block count 9.  This validates D16 semantics and cleanup only; it is
not a larger-D speed or memory-scaling result.

## Local file identities

SHA256 at validation:

- engine `Project.toml`: `02c8ed305e155a7f116f945fdfde89e0310fbd3b09215e1b4473c823a89105b6`
- engine `Manifest.toml`: `976211682818c8a51ba727b49b2e315df9854961100843bf54b835f30c58cfbd`
- converter `Project.toml`: `ab74c200af242289610297fe9a774efd9967fd8b7016c9f587440aa2d520b3dc`
- converter `Manifest.toml`: `0a2bd2c5f7d18c70ea897b5403ec12c070a89956873f6b363933ac1f1995f5ef`
- `environment_smoke.jl`: `6d41cb84ff13728d88629862daf824f979b694c37fef766269f0a5a4713fdd7a`
- canonical operator helper after density-operator additions: `0542b28666cfa2909b5cbee88b8a388c47ca4df0d9f03561ce7103146f7614fa`
- U1xU1 model: `e2ef4b620cadf47fcea2c484aea62eaedf7a133d3c3cb42e8d7420c3630e927b`
- tensor helper/API repair: `0df851d5e10e32b546e7c9a8b7718df13c2613f4b36c5d1af29e8dc0b717ccbf`
- operator regression: `77b1eca2e66ca7b725b51e5bc866813febce10ac6c58d5b13280f6458a6b82a0`
- 2x2 model smoke: `5c0470768ed589b5a7c347f10eef216f37823959bf7ed6839e4df74730964455`
- ITensor cross-check: `ca8d76973249ac645c77718edf175f4c6dd0592c9d7cdd393be74fd392e68e31`
- target adapter core: `1d17b039ea78a3d81bd588e8989e855a60386848488862dbea164504afbf38d8`
- target engine wrapper: `931713bfdf653dc92fd5e383f7382282e0587e547e2882ba1973621223d8a49d`
- independent ED reference: `7092adaa25549dd7c97bcd36ceb4694c74715ade9f41747e78eb9ae39d098ebb`
- adapter/ED test: `52a2ab6c0c0fc0156afe3e13ba96999f75ad912c6d7110dc195825e3afdd243e`
- U12 ITensor dense test: `9a5dd04fcfedfc4d26bfaf044edd49796aadcecd992af4f4a84f3e15532edb60`
- same-D ITensor driver: `d57033d2d3182f1fbca8cfda294ce798251f112ce93f5aea703ed583a9f570a1`
- final `MPS.jl`: `442c50448b2741298e02493ff445fc1140356b15c56adb6c7a9d6202f3d53fc7`
- rolling-store `MPS/TN.jl`: `ea182f414995d0e6ae33725f0a234228779c1844fc8785c32c4c13fff50ebd97`
- rolling-store `MPS.jl`: `64e27b0d535380645fd5d95a60a3ebe9b5579d467f8b7c263c28d55f0f5cd573`
- rolling-store `start.jl`: `5dc2382c54669c16a8e45e4c32064ae68c7553e0369853df1543f764b1d34cc8`
- rolling environment MPI unit: `70f26c6299f31d63450d5f94c9149e179e94598fda4136e305b5607f3a4c58c5`
- rolling D16 gate: `ff785b3c97a8748fefdac3046a9ce28cb57f659a1009260b1769ffc8b517457d`
- final `MPS/TN.jl`: `3091f9211429a0c8aea2d3ea1e0fd3cd83f3f7197893f76b32f2ed0b06f8acb0`
- final `start.jl`: `7fb6f944680fafff9444040b87663dcf1bf62c6e16fb80144fd239bfe890cd93`
- final target wrapper: `9a24d2d7055b91c44a0d049c435eac82122d0f5a2a6bacd1b5ef488722f3c53b`
- D1000 round-trip comparator: `6396db13ef7cdefd5819f692997c9507ff0e41fdd7b4c6c0852e024e941b7aa4`
- D1000 TensorKit Hamiltonian measurement: `e4d30014829dcb35fbd5ea2ba903549903934e93ef3546869819f35afce58f50`
