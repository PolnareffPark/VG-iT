#!/bin/bash
# ==========================================================
# VG Generality Experiment: SF VG-iFlashformer
#
# Shows Variable Grouping (with Shifted Grouping + FiLM) combined
# with SDPA kernel — VG as a wrapper is attention-kernel-agnostic.
#
# 9 datasets x 4 PL x 5 seeds x 1 model = 180 runs
#
# Usage:
#   bash scripts/run_vg_generality.sh
#   RESUME=1 bash scripts/run_vg_generality.sh
#   DEBUG=1 bash scripts/run_vg_generality.sh
# ==========================================================

set -eo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESUME="${RESUME:-1}"

# Debug mode
if [ "${DEBUG:-0}" == "1" ]; then
    echo "!!! DEBUG MODE: 2 epochs !!!"
    train_args=(--train_epochs 2 --patience 1 --num_workers 0)
else
    train_args=(--train_epochs 100 --patience 5 --num_workers 4)
fi

# Precision (BF16 for Ampere GPUs)
precision_args=(--use_amp --amp_dtype bf16)
compile_args=(--use_compile)

# Common
seq_len=96
d_model=512

# Output
results_dir="./test_results/vg_generality"
done_dir="$results_dir/.done"
failed_dir="$results_dir/.failed"
summary_file="./test_results/vg_generality_results.csv"
oom_skip_file="$results_dir/.oom_skip"
mkdir -p "$results_dir" "$done_dir" "$failed_dir"

pred_lens=(96 192 336 720)
seeds=(2021 2022 2023 2024 2025)

# Model: SF VG-iFlashformer (VG + Shifted Grouping + FiLM + SDPA)
model_name="VG_iFlashformer"

# Dataset configs (unified: benchmark + industrial)
# Format: "name|root_path|data_path|freq|enc_in|e_layers|d_ff|lr|batch_size|data_flag|num_groups"
#
# Hyperparameters: baseline settings per dataset (same as iTransformer/iFlashformer/iNystromformer)
# G (num_groups): per-dataset sqrt(N) rule — 2^round(log2(sqrt(N)))
datasets=(
    # Benchmark (3)
    "traffic|./dataset/traffic/|traffic.csv|h|862|4|512|0.001|16|custom|32"
    "electricity|./dataset/electricity/|electricity.csv|h|321|3|512|0.0005|16|custom|16"
    "solar_AL|./dataset/Solar/|solar_AL.txt|t|137|2|512|0.0005|32|Solar|16"
    # Industrial (6)
    "basf|./dataset/basf/|basf.csv|h|244|3|512|0.0005|16|custom|16"
    "kamp|./dataset/kamp/|kamp.csv|t|228|3|512|0.0005|16|custom|16"
    "bdg2|./dataset/bdg2/|bdg2.csv|h|2817|3|512|0.0005|8|custom|64"
    "ashrae|./dataset/ashrae_energy/|ashrae.csv|h|2362|3|512|0.0005|8|custom|64"
    "sdwpf|./dataset/sdwpf_energy/|sdwpf.csv|t|2144|3|512|0.0005|8|custom|64"
    "care_wind|./dataset/care_wind_energy/|care_wind.csv|t|238|3|512|0.0005|16|custom|16"
)

# ========== Helpers ==========
format_duration() {
    local seconds=$1
    local h=$((seconds / 3600))
    local m=$(( (seconds % 3600) / 60 ))
    local s=$((seconds % 60))
    if [ $h -gt 0 ]; then
        printf "%dh %02dm %02ds" $h $m $s
    else
        printf "%dm %02ds" $m $s
    fi
}

# ========== Count total runs (accounting for RESUME skips) ==========
total_runs=0
skipped_runs=0
jobs=()

prefix="vgflash"
for dataset_entry in "${datasets[@]}"; do
    IFS='|' read -r ds_name root_path data_path freq enc_in e_layers d_ff lr batch_size data_flag num_groups <<< "$dataset_entry"

    for pl in "${pred_lens[@]}"; do
        for seed in "${seeds[@]}"; do
            total_runs=$((total_runs + 1))
            job_id="${prefix}_${ds_name}_pl${pl}_s${seed}"

            if [ "$RESUME" == "1" ] && [ -f "$done_dir/$job_id" ]; then
                skipped_runs=$((skipped_runs + 1))
            else
                rm -f "$failed_dir/$job_id" 2>/dev/null || true
                jobs+=("${job_id}|${ds_name}|${root_path}|${data_path}|${freq}|${enc_in}|${e_layers}|${d_ff}|${lr}|${batch_size}|${data_flag}|${num_groups}|${pl}|${seed}")
            fi
        done
    done
done

remaining=${#jobs[@]}

echo "=========================================="
echo "VG Generality Experiment (SF VG-iFlashformer)"
echo "=========================================="
echo "Model             : $model_name"
echo "GPU               : $CUDA_VISIBLE_DEVICES"
echo "Total runs        : $total_runs"
echo "Already done      : $skipped_runs"
echo "Remaining         : $remaining"
echo "Resume            : $RESUME"
echo "=========================================="

if [ "$remaining" -eq 0 ]; then
    echo "All runs already completed. Nothing to do."
    exit 0
fi

# ========== Run experiments sequentially ==========
global_start=$(date +%s)
processed=0
done_count=0
fail_count=0
oom_count=0

for job_entry in "${jobs[@]}"; do
    IFS='|' read -r job_id ds_name root_path data_path freq enc_in e_layers d_ff lr batch_size data_flag num_groups pl seed <<< "$job_entry"

    processed=$((processed + 1))

    # OOM skip: same (dataset, pred_len) already OOMed
    oom_key="${ds_name}_pl${pl}"
    if grep -qF "$oom_key" "$oom_skip_file" 2>/dev/null; then
        echo "$model_name,$ds_name,$enc_in,$pl,$seed,OOM_SKIP,OOM_SKIP,$freq,$job_id" > "$done_dir/$job_id"
        oom_count=$((oom_count + 1))
        echo "[${processed}/${remaining}] $(date '+%H:%M:%S') SKIP   $job_id  (OOM on same config)"
        continue
    fi

    log_file="$results_dir/${job_id}.log"
    job_start=$(date +%s)

    echo ""
    echo "=========================================================="
    echo "[${processed}/${remaining}] $(date '+%H:%M:%S') START: $job_id"
    echo "  Dataset=$ds_name  PL=$pl  Seed=$seed  G=$num_groups"
    echo "=========================================================="

    # SF VG-iFlashformer args
    model_args=(
        --model "$model_name"
        --num_groups "$num_groups"
        --pooling mean
        --use_shifted_grouping 1
        --use_film_broadcast 1
        --use_global_interact 1
    )

    if PYTHONUNBUFFERED=1 python run.py \
        --is_training 1 \
        --data "$data_flag" \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$job_id" \
        "${model_args[@]}" \
        --features M \
        --freq "$freq" \
        --pred_len "$pl" \
        --seq_len "$seq_len" \
        --seed "$seed" \
        --e_layers "$e_layers" \
        --d_model "$d_model" \
        --d_ff "$d_ff" \
        --batch_size "$batch_size" \
        --learning_rate "$lr" \
        --enc_in "$enc_in" \
        --dec_in "$enc_in" \
        --c_out "$enc_in" \
        --output_subdir vg_generality \
        --itr 1 \
        "${precision_args[@]}" \
        "${compile_args[@]}" \
        "${train_args[@]}" \
        --skip_flops_profiling 1 \
        > "$log_file" 2>&1; then

        mse=$(grep -oP "mse:\K[0-9.]+" "$log_file" 2>/dev/null | tail -1 || true)
        [ -z "$mse" ] && mse="N/A"
        mae=$(grep -oP "mae:\K[0-9.]+" "$log_file" 2>/dev/null | tail -1 || true)
        [ -z "$mae" ] && mae="N/A"

        echo "$model_name,$ds_name,$enc_in,$pl,$seed,$mse,$mae,$freq,$job_id" > "$done_dir/$job_id"
        done_count=$((done_count + 1))

        elapsed=$(( $(date +%s) - job_start ))
        echo "[${processed}/${remaining}] $(date '+%H:%M:%S') DONE   $job_id ($(format_duration $elapsed)) MSE=${mse}"
    else
        exit_code=$?
        elapsed=$(( $(date +%s) - job_start ))

        if grep -q "CUDA out of memory\|OutOfMemoryError" "$log_file" 2>/dev/null; then
            echo "$model_name,$ds_name,$enc_in,$pl,$seed,OOM,OOM,$freq,$job_id" > "$done_dir/$job_id"
            oom_count=$((oom_count + 1))
            echo "[${processed}/${remaining}] $(date '+%H:%M:%S') OOM    $job_id ($(format_duration $elapsed)) CUDA OOM"
            echo "$oom_key" >> "$oom_skip_file"
        elif [ "$exit_code" -eq 137 ] || grep -q "MemoryError\|Cannot allocate memory" "$log_file" 2>/dev/null; then
            echo "$model_name,$ds_name,$enc_in,$pl,$seed,RAM_OOM,RAM_OOM,$freq,$job_id" > "$done_dir/$job_id"
            oom_count=$((oom_count + 1))
            echo "[${processed}/${remaining}] $(date '+%H:%M:%S') OOM    $job_id ($(format_duration $elapsed)) RAM OOM (exit=$exit_code)"
            echo "$oom_key" >> "$oom_skip_file"
        else
            touch "$failed_dir/$job_id"
            fail_count=$((fail_count + 1))
            echo "[${processed}/${remaining}] $(date '+%H:%M:%S') FAILED $job_id ($(format_duration $elapsed)) — see $log_file"
        fi
    fi

    # ETA
    total_elapsed=$(( $(date +%s) - global_start ))
    completed=$((done_count + fail_count + oom_count))
    if [ $completed -gt 0 ]; then
        eta_sec=$(( (remaining - processed) * total_elapsed / completed ))
        echo "  [PROGRESS] ${processed}/${remaining} | Done: ${done_count} | OOM: ${oom_count} | Failed: ${fail_count} | ETA: $(format_duration $eta_sec)"
    fi
done

# ========== Merge Results ==========
echo ""
echo "Merging results..."
echo "model,dataset,enc_in,pred_len,seed,MSE,MAE,freq,job_id" > "$summary_file"

if [ "$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    for f in "$done_dir"/*; do
        [ -f "$f" ] || continue
        cat "$f"
    done | sort -t, -k1,1 -k2,2 -k4,4n -k5,5n >> "$summary_file"
fi

# ========== Final Report ==========
global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
done_total=$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
fail_total=$(find "$failed_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "SF VG-iFlashformer Experiment Complete!"
echo "=========================================="
echo "Total: $total_runs | Done: $done_total | Skipped: $skipped_runs | Failed: $fail_total"
echo "Wall-clock: $(format_duration $global_elapsed)"
echo "Summary: $summary_file"
echo "=========================================="

if [ "$fail_total" -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    for f in "$failed_dir"/*; do [ -f "$f" ] && echo "  - $(basename "$f")"; done
    exit 1
fi
