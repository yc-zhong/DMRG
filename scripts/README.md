# DMRG scripts

- `slurm/`: active batch entry points.
- `submit/`: submission wrappers and dependency wiring.
- `legacy/`: retained historical entry points; do not use as templates for new
  production work.

Component-specific launchers stay with their component, such as `converter/`
and `test/`. Reusable numerical logic belongs in the Julia modules, not here.
