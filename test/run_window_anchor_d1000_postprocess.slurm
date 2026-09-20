#!/usr/bin/env bash
#SBATCH --job-name=tk_anchor_d1k_post
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_d1000_benchmark/postprocess_logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_anchor_d1000_benchmark/postprocess_logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v2_20260918"
readonly converter_root="${migration_root}/converter_anchor_v2_20260918"
readonly source_root="${migration_root}/window_anchor_d1000_benchmark/run_12167401"
readonly post_root="${source_root}/postprocess_${SLURM_JOB_ID}"

test ! -e "${post_root}"
mkdir -p "${post_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${engine_root}"
for file in MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl start.jl; do
  expected=$(awk -v file="${file}" '$2==file {print $1}' "${source_root}/engine_sha256.txt")
  test -n "${expected}"
  actual=$(sha256sum "${file}" | awk '{print $1}')
  test "${actual}" = "${expected}"
done
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl start.jl \
  >"${post_root}/engine_core_sha256.txt"
sha256sum "${converter_root}/compare_tensorkit_mps.jl" \
  >"${post_root}/converter_sha256.txt"
cmp -s "${post_root}/converter_sha256.txt" "${source_root}/converter_sha256.txt"
cp "${source_root}/source_CURRENT" "${post_root}/source_CURRENT"

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}

for spec in 'legacy_a false 660 0 0' 'anchor_a true 343 5 5' \
    'anchor_b true 343 5 5' 'legacy_b false 660 0 0'; do
  set -- ${spec}
  label=$1 anchors=$2 updates=$3 snapshots=$4 restores=$5
  output="${source_root}/${label}.out"
  checkpoint="${source_root}/${label}_mps"
  test -s "${output}" && test -s "${source_root}/${label}.err"
  test -f "${checkpoint}/CURRENT" && test -f "${checkpoint}/PREVIOUS"
  generation=$(current_generation "${checkpoint}")
  test -f "${generation}/checkpoint_manifest.toml"
  grep -Eq '^complete = true$' "${generation}/checkpoint_manifest.toml"
  test "$(find "${generation}" -maxdepth 1 -type f -name '[0-9]*.jld2' | wc -l)" -eq 192
  grep -Eq "WINDOW_WORK direction=forward.*window=32 anchors=${anchors}.*prepare_calls=6.*environment_site_updates=${updates}.*anchor_snapshots=${snapshots}.*anchor_restores=${restores}" "${output}"
  grep -Eq 'drops = [1-9][0-9]*' "${output}"
  ! grep -Eq 'missing_drops = [1-9][0-9]*' "${output}"

  monitor="${source_root}/${label}.storage.txt"
  valid=$(awk -F'[ =]' '/^# monitor_summary/ {for(i=1;i<=NF;i++)if($i=="valid_samples")print $(i+1)}' "${monitor}")
  failed=$(awk -F'[ =]' '/^# monitor_summary/ {for(i=1;i<=NF;i++)if($i=="failed_samples")print $(i+1)}' "${monitor}")
  test -n "${valid}" && test "${valid}" -ge 10
  test -n "${failed}" && test "${failed}" -ge 0
  printf '%s %s %s\n' "${label}" "${valid}" "${failed}" >>"${post_root}/monitor_summary.tsv"
  for column in '2 primary' '3 spill' '4 combined'; do
    set -- ${column}
    awk -v col="$1" 'NR>1 && $1 ~ /^[0-9]+$/ && $col>p {p=$col} END {print p+0}' \
      "${monitor}" >"${post_root}/${label}.$2_peak_kib"
  done
done

# The original batch generated performance.tsv only after every run_case had
# executed verify_cleanup on its execution node.  Node-local /dev/shm and /tmp
# paths cannot be rechecked independently from this later postprocess node.
test -s "${source_root}/performance.tsv"
printf '%s\n' \
  'cleanup_evidence=original run_case verify_cleanup passed before performance.tsv was generated' \
  >"${post_root}/cleanup_provenance.txt"

printf 'case anchors case_wall_s sweep_s window_prepare_s anchor_snapshot_s anchor_restore_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib combined_peak_kib\n' \
  >"${post_root}/performance_corrected.tsv"
for label in legacy_a anchor_a anchor_b legacy_b; do
  raw=$(awk -v label="${label}" '$1==label {print}' "${source_root}/performance.tsv")
  test -n "${raw}"
  set -- ${raw}
  test "$#" -eq 18
  printf '%s' "$1" >>"${post_root}/performance_corrected.tsv"
  for field in "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
      "${10}" "${11}" "${12}" "${13}" "${14}" "${15}"; do
    printf ' %s' "${field}" >>"${post_root}/performance_corrected.tsv"
  done
  printf ' %s %s %s\n' \
    "$(cat "${post_root}/${label}.primary_peak_kib")" \
    "$(cat "${post_root}/${label}.spill_peak_kib")" \
    "$(cat "${post_root}/${label}.combined_peak_kib")" \
    >>"${post_root}/performance_corrected.tsv"
done

awk '
  function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
  NR==1 {next}
  {
    if(NF!=18 || ($2!="true" && $2!="false")) bad=1
    for(i=3;i<=18;i++) if(!finite($i)) bad=1
    for(i=3;i<=15;i++) if(i!=6 && i!=7 && $i<=0) bad=1
    if($16<=0 || $17<0 || $18<=0) bad=1
    count[$2]++
  }
  END {exit !(bad==0 && count["true"]==2 && count["false"]==2)}' \
  "${post_root}/performance_corrected.tsv"

compare_pair() {
  local candidate=$1
  local result="${post_root}/legacy_a_vs_${candidate}.txt"
  cd "${converter_root}"
  julia --project=. compare_tensorkit_mps.jl \
    "$(current_generation "${source_root}/legacy_a_mps")" \
    "$(current_generation "${source_root}/${candidate}_mps")" | tee "${result}"
  awk -F= '
    function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
    $1=="normalized_overlap" {overlap=$2; so=1}
    $1=="fidelity_per_site" {fidelity=$2; sf=1}
    $1=="bra_norm2" {bra=$2; sb=1}
    $1=="ket_norm2" {ket=$2; sk=1}
    $1=="site_spaces_match" {spaces=$2; ss=1}
    $1=="bra_max_bond_dimension" {bd=$2; sbd=1}
    $1=="ket_max_bond_dimension" {kd=$2; skd=1}
    $1=="bra_max_block_count" {bb=$2; sbb=1}
    $1=="ket_max_block_count" {kb=$2; skb=1}
    END {
      if(!(so&&sf&&sb&&sk&&ss&&sbd&&skd&&sbb&&skb))exit 1
      if(!(finite(overlap)&&finite(fidelity)&&finite(bra)&&finite(ket)))exit 1
      if(overlap<1-1e-10 || overlap>1+1e-10)exit 1
      if(fidelity<1-1e-12 || fidelity>1+1e-12)exit 1
      d=bra-ket;if(d<0)d=-d
      if(!(d<=1e-10 && bra>=1-1e-10 && bra<=1+1e-10 && ket>=1-1e-10 && ket<=1+1e-10))exit 1
      exit !(spaces=="true" && bd==kd && bb==kb)
    }' "${result}"
}

for candidate in anchor_a anchor_b legacy_b; do
  compare_pair "${candidate}"
  paste <(awk -F, 'NR>1 {print $3,$4}' "${source_root}/legacy_a.out_short.csv") \
        <(awk -F, 'NR>1 {print $3,$4}' "${source_root}/${candidate}.out_short.csv") | \
    awk 'function finite(x) {return x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/}
      {seen++;if(NF!=4)bad=1;for(i=1;i<=4;i++)if(!finite($i))bad=1;dt=$1-$3;if(dt<0)dt=-dt;de=$2-$4;if(de<0)de=-de;s=($1<0?-$1:$1);q=($3<0?-$3:$3);if(q>s)s=q;t=1e-12;if(1e-8*s>t)t=1e-8*s;if(dt>t || de>1e-12)bad=1}
      END {exit !(seen==1 && bad==0)}'
done
printf 'numerical=PASS\n' >"${post_root}/NUMERICAL_PASS"

awk 'NR>1 {kind=($2=="true"?"anchor":"legacy"); n[kind]++; for(i=3;i<=18;i++)s[kind,i]+=$i}
  END {print "kind case_wall_s sweep_s window_prepare_s anchor_snapshot_s anchor_restore_s h_action_s krylov_s svd_s block_update_s checkpoint_prepare_s checkpoint_finalize_s sum_rank_hwm_kib max_rank_hwm_kib primary_peak_kib spill_peak_kib combined_peak_kib";
       for(k in n){printf "%s",k;for(i=3;i<=18;i++)printf " %.9g",s[k,i]/n[k];printf "\n"}}' \
  "${post_root}/performance_corrected.tsv" >"${post_root}/performance_mean.tsv"

metric() { awk -v kind=$1 -v col=$2 '$1==kind {print $col;found=1} END {if(!found)exit 1}' "${post_root}/performance_mean.tsv"; }
max_metric() { awk -v anchors=$1 -v col=$2 'NR>1 && $2==anchors {if(!found || $col>value)value=$col;found=1} END {if(!found)exit 1;print value}' "${post_root}/performance_corrected.tsv"; }

legacy_wall=$(metric legacy 2); anchor_wall=$(metric anchor 2)
legacy_sweep=$(metric legacy 3); anchor_sweep=$(metric anchor 3)
legacy_env=$(metric legacy 4); anchor_env=$(metric anchor 4)
awk -v old="${legacy_env}" -v new="${anchor_env}" 'BEGIN {exit !(new<=0.75*old)}'
awk -v old="${legacy_sweep}" -v new="${anchor_sweep}" 'BEGIN {exit !(new<=0.92*old)}'
awk -v old="${legacy_wall}" -v new="${anchor_wall}" 'BEGIN {exit !(new<=0.92*old)}'
for col in 7 8 9; do
  awk -v old="$(metric legacy "${col}")" -v new="$(metric anchor "${col}")" 'BEGIN {exit !(new<=1.05*old)}'
done
legacy_backing_max=$(max_metric false 18); anchor_backing_max=$(max_metric true 18)
legacy_rss_max=$(max_metric false 14); anchor_rss_max=$(max_metric true 14)
anchor_spill_max=$(max_metric true 17)
awk -v old="${legacy_backing_max}" -v new="${anchor_backing_max}" 'BEGIN {exit !(new<=1.25*old)}'
awk -v old="${legacy_rss_max}" -v new="${anchor_rss_max}" 'BEGIN {exit !(new<=1.10*old)}'
awk -v spill="${anchor_spill_max}" 'BEGIN {exit !(spill<=4)}'

{
  printf 'metric legacy_mean anchor_mean anchor_over_legacy speedup\n'
  for spec in 'case_wall 2' 'sweep_total 3' 'window_prepare 4' 'h_action 7' 'krylov 8' 'svd 9' 'aggregate_rss_kib 13' 'combined_backing_kib 17'; do
    set -- ${spec}; old=$(metric legacy "$2"); new=$(metric anchor "$2")
    awk -v name="$1" -v old="${old}" -v new="${new}" 'BEGIN {printf "%s %.9g %.9g %.9g %.9g\n",name,old,new,new/old,old/new}'
  done
} >"${post_root}/speedup.tsv"
{
  printf 'kind max_sum_rank_hwm_kib max_combined_backing_kib max_spill_kib\n'
  printf 'legacy %s %s %s\n' "${legacy_rss_max}" "${legacy_backing_max}" "$(max_metric false 17)"
  printf 'anchor %s %s %s\n' "${anchor_rss_max}" "${anchor_backing_max}" "${anchor_spill_max}"
} >"${post_root}/capacity_max.tsv"

printf 'performance=PASS\n' >"${post_root}/PERFORMANCE_PASS"
printf 'benchmark=COMPLETE\n' >"${post_root}/BENCHMARK_COMPLETE"
printf 'd1000_anchor_acceptance=PASS\n' | tee "${post_root}/PASS"
