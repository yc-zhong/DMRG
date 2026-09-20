# ITensor to TensorKit MPS bridge

This directory contains an isolated compatibility prototype. It does not alter
the source HDF5 checkpoint and refuses to write into a non-empty destination.

The local basis is the ITensor `Electron` order
`[Emp, Up, Dn, UpDn]`. The two conserved labels are particle number `Nf` and
integer spin label `Nup-Ndn`. ITensor suffix charges are converted to TensorKit
prefix charges as `q_prefix = q_total - q_suffix`; the MPS site order is not
changed.

Instantiate with Julia 1.11 on the cluster before production use:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Convert and validate a small checkpoint:

```sh
julia --project=. itensor_to_tensorkit.jl SOURCE.h5 NEW_OUTPUT_DIR
julia --project=. validate_roundtrip.jl SOURCE.h5 NEW_OUTPUT_DIR
```

Compare two TensorKit checkpoints directly, without forming the exponentially
large dense many-body state:

```sh
julia --project=. compare_tensorkit_mps.jl BRA_DIR KET_DIR
```

This reports both norms, the normalized overlap, a per-site fidelity, maximum
bond dimensions, and maximum QN-block counts.  It supports different virtual
bond dimensions and block multiplicities, provided the two MPS use the same
physical space and site ordering.

Measure the norm, conserved totals, local density/spin/double occupancy, and
selected connected charge/spin correlators directly in engine format:

```sh
julia --project=. measure_tensorkit_hubbard.jl MPS_DIR '1:2,1:192,96:97'
```

To make `validate_roundtrip.jl` also compare the energy under the target
OBC x OBC Hubbard Hamiltonian with single-left-edge pinning (whose coefficient
multiplies `Nup-Ndn`), append `Lx Ly U h`, for example `16 2 6.0 0.1`.

The implementation uses a dense work array for one site at a time. That is a
deliberate first-stage simplification. A sector-by-sector reader should replace
it before attempting D=20000 conversion.
