#!/usr/bin/env bash
#SBATCH --job-name=tk_hybrid_aff
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_thread_affinity_smoke/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_thread_affinity_smoke/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root=${migration_root}/engine_leftedge_haction_v6_20260919
readonly test_root=${migration_root}/hybrid_thread_affinity_smoke
readonly driver=${test_root}/hybrid_thread_affinity_smoke.jl
readonly batch_script=${test_root}/run_hybrid_thread_affinity_smoke.slurm
readonly run_root=${test_root}/run_${SLURM_JOB_ID}

test ! -e "${run_root}"
mkdir -p "${run_root}"

export JULIA_DEPOT_PATH=${migration_root}/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export TENSORKIT_TRANSFORMER_THREADS=1
export OMP_THREADS=16 OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export OMP_PLACES=cores OMP_PROC_BIND=close
export MKL_DYNAMIC=false
export SMOKE_GEMM_N=2048
export AFFINITY_RESULT_ROOT=${run_root}

sha256sum "${driver}" "${batch_script}" >"${run_root}/sha256.txt"
{
  printf 'job_id=%s\nranks=4\ncpus_per_rank=16\n' "${SLURM_JOB_ID}"
  printf 'julia_threads=1\ntensorkit_transformer_threads=1\nblas_threads=16\n'
  printf 'binding=slurm_cpu_bind_cores\n'
} >"${run_root}/manifest.txt"

cd "${run_root}"
/usr/bin/time -v srun --mpi=pmi2 --ntasks=4 --cpus-per-task=16 \
  --cpu-bind=cores julia --project="${engine_root}" "${driver}" \
  >"${run_root}/engine.out" 2>"${run_root}/engine.err"

grep -q '^hybrid_thread_affinity_smoke=PASS ranks=4$' "${run_root}/engine.out"
test "$(find "${run_root}" -maxdepth 1 -name 'rank_*.txt' -type f | wc -l | awk '{print $1}')" -eq 4
cat "${run_root}"/rank_*.txt | grep '^HYBRID_AFFINITY rank=' | sort >"${run_root}/affinity.tsv"
cat "${run_root}"/rank_*.txt | grep '^HYBRID_THREAD_AFFINITY rank=' | sort \
  >"${run_root}/thread_affinity.tsv"
test "$(cat "${run_root}"/rank_*.txt | grep -c '^HYBRID_GEMM rank=')" -eq 4
awk '
  {
    julia=""; transformer=""; blas=""
    for(i=1;i<=NF;i++) {
      split($i,p,"=")
      if(p[1]=="julia_threads") julia=p[2]
      if(p[1]=="transformer_threads") transformer=p[2]
      if(p[1]=="blas_threads") blas=p[2]
    }
    if(julia!=1 || transformer!=1 || blas!=16) exit 3
  }
  END {if(NR!=4) exit 4}
' "${run_root}/affinity.tsv"
awk '
  {
    rank=""; cpu=""
    for(i=1;i<=NF;i++) {
      split($i,p,"=")
      if(p[1]=="rank") rank=p[2]
      if(p[1]=="cpus_allowed") cpu=p[2]
    }
    if(rank=="" || cpu=="") exit 2
    nranges=split(cpu,ranges,",")
    for(r=1;r<=nranges;r++) {
      nparts=split(ranges[r],ends,"-")
      lo=ends[1]+0; hi=(nparts==1 ? lo : ends[2]+0)
      if(hi<lo) exit 5
      for(c=lo;c<=hi;c++) {
        rank_cpu[rank SUBSEP c]=1
      }
    }
  }
  END {
    for(r=0;r<4;r++) {
      count=0
      for(c=0;c<4096;c++) if(rank_cpu[r SUBSEP c]) {
        count++
        if(global_cpu[c]++) exit 6
      }
      if(count!=16) exit 7
    }
  }
' "${run_root}/thread_affinity.tsv"

printf 'hybrid_thread_affinity_smoke=PASS\n' >"${run_root}/PASS"
