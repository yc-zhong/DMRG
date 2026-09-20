# Hubbard DMRG Project Guide

This directory contains the active repulsive-Hubbard DMRG work. Reuse and
extend the existing model, measurement, checkpoint, and continuation code;
keep run scripts focused on parameters and package calls.

## Development direction

- Prefer the newer DMRG implementation for active development. Treat the old
  ITensor code as a source of validated conventions and useful utilities, not
  as a second implementation that must be maintained in parallel.
- Port or wrap only legacy pieces that are relevant to current work, such as
  checkpoint conversion, measurements, exact benchmarks, and restart tools.
  Unrelated historical scripts may remain untouched.
- Put reusable Hamiltonian construction, solver stages, observables,
  checkpoint I/O, conversion, and scan logic behind the canonical modules.
  Slurm and run scripts should mainly choose parameters, call those APIs, and
  select unique output locations.
- Correct or extend the canonical implementation instead of adding `new`,
  `fixed`, `v2`, or `final` variants. Preserve an old entry point only when a
  recorded result or restart path depends on it.

## Physical and numerical conventions

- Preserve the Hubbard model, site ordering, filling sector, and flux convention
  unless the task explicitly changes them.
- Production pinning is edge-only staggered AF. Keep `edge_af` (both x ends)
  distinct from `left_edge_af` (left end only), and never pool or overwrite the
  two datasets.
- Through `D=4000`, use the established split-MPO policy with `krylovdim=3` and
  `maxiter=1`. At larger `D`, use the combined MPO with `krylovdim=5`,
  `maxiter=2`, and block-sparse contraction threading when memory requires it.
- Assess a stage using energy change, maximum truncation error, and checkpoint
  metadata. A small truncation error alone is not an energy-convergence test.
- Run the exact `U=0` ED benchmark before interpreting a flux scan.

## Outputs and provenance

- Keep rolling MPS checkpoints and retain the previous verified checkpoint until
  its replacement has been opened and checked.
- Use unique output and MPS directories for continuations; do not overwrite or
  combine unrelated pinning geometries.
- Record geometry, `U`, filling, gauge, pinning, bond-dimension schedule, sweep
  policy, cutoff, Krylov settings, checkpoint, and job ID with reported results.
- Cluster work uses `/public4/home/sc56578/zyc/hubbard_flux` through the `bscc`
  SSH alias.
