#!/bin/bash
# ==========================================================
# Grouping Invariance Experiment (Exp 4) — Shard-based Distributed
#
# Purpose: Test grouping method invariance with 30 seeds across
#   all 8 methods on 3 benchmark datasets.
#
# Default design: 8 methods × 3 datasets × 4 PL × 30 seeds = 2,880 runs
# Shard-based: split across multiple GPUs/servers.
#
# Methods:
#   ordered, random, finch_like, coarsening, mi_based,
#   anti_clustering, score_stratified, maximin_dispersion
#
# Usage:
#   SHARD=0  TOTAL_SHARDS=12 bash scripts/run_grouping_invariance.sh  # GPU 0
#   SHARD=1  TOTAL_SHARDS=12 bash scripts/run_grouping_invariance.sh  # GPU 1
#   ...
#   SHARD=11 TOTAL_SHARDS=12 bash scripts/run_grouping_invariance.sh  # GPU 11
#
# Single-GPU (all 2,880):
#   bash scripts/run_grouping_invariance.sh
#
# Options:
#   GPU=0             CUDA device index (default: 0)
#   SHARD=0           Shard index (default: 0)
#   TOTAL_SHARDS=1    Total number of shards (default: 1 = no sharding)
#   RESUME=1          Skip completed jobs (default: 1)
#   DEBUG=1           2 epochs for smoke testing
#   METHODS=ordered   Optional comma-separated method subset
#   SEEDS=2021,2022   Optional comma-separated seed subset
#
# Graceful stop:
#   touch ./test_results/grouping_invariance/.stop
# ==========================================================

set -eo pipefail

# ========== Configuration ==========

GPU="${GPU:-0}"
SHARD="${SHARD:-0}"
TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
RESUME="${RESUME:-1}"

if [ "${DEBUG:-0}" == "1" ]; then
    echo "!!! DEBUG MODE: 2 epochs !!!"
    train_args=(--train_epochs 2 --patience 1 --num_workers 0)
else
    train_args=(--train_epochs 100 --patience 5 --num_workers 4)
fi

export CUDA_VISIBLE_DEVICES=$GPU

# Fixed model parameters
pooling=mean
seq_len=96
d_model=512
global_interact=1

# Precision (bf16)
precision_args=(--use_amp --amp_dtype bf16)
compile_args=(--use_compile)

# SF flags
sf_args=(--use_shifted_grouping 1 --use_film_broadcast 1)

# Grouping methods. The optional filter keeps the same provenance-generating code path
# for the paper's ordered 2021-2025 main subset.
if [ -n "${METHODS:-}" ]; then
    IFS=',' read -ra methods <<< "$METHODS"
else
    methods=(ordered random finch_like coarsening mi_based anti_clustering score_stratified maximin_dispersion)
fi

# Datasets: "name|root_path|data_path|N|e_layers|d_ff|lr|batch_size|data_flag|enc_in|num_groups"
# num_groups: per-dataset G = 2^round(log2(sqrt(N)))
datasets=(
    "traffic|./dataset/traffic/|traffic.csv|862|4|512|0.001|16|custom|862|32"
    "electricity|./dataset/electricity/|electricity.csv|321|3|512|0.0005|16|custom|321|16"
    "solar_AL|./dataset/Solar/|solar_AL.txt|137|2|512|0.0005|32|Solar|137|16"
)

pred_lens=(96 192 336 720)

# Seeds: 2021-2050 (30 seeds) by default; optionally restrict without changing
# model, dataset, grouping, precision, or optimization settings.
if [ -n "${SEEDS:-}" ]; then
    IFS=',' read -ra seeds <<< "$SEEDS"
else
    seeds=()
    for s in $(seq 2021 2050); do
        seeds+=($s)
    done
fi

# Directories
results_dir="./test_results/grouping_invariance"
done_dir="$results_dir/.done"
failed_dir="$results_dir/.failed"
stop_file="$results_dir/.stop"
summary_file="./test_results/grouping_invariance_results.csv"

progress_file="$results_dir/.progress_shard${SHARD}"

oom_skip_file="$results_dir/.oom_skip"

mkdir -p "$results_dir" "$done_dir" "$failed_dir"
# Only clear stop file if this is shard 0
if [ "$SHARD" -eq 0 ]; then
    rm -f "$stop_file"
fi

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

monitor_progress() {
    local start_time=$1
    local initial_done=$2
    local initial_fail=$3
    local batch_total=$4

    while true; do
        sleep 30

        local done_count fail_count
        done_count=$(find "$done_dir" -maxdepth 1 -name "gi_*" -type f 2>/dev/null | wc -l)
        fail_count=$(find "$failed_dir" -maxdepth 1 -name "gi_*" -type f 2>/dev/null | wc -l)
        local new_done=$((done_count - initial_done))
        local new_fail=$((fail_count - initial_fail))
        local processed=$((new_done + new_fail))

        local elapsed=$(( $(date +%s) - start_time ))
        local eta_str="calculating..."

        if [ "$new_done" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
            local remaining_jobs=$((batch_total - processed))
            local eta_sec=$(( remaining_jobs * elapsed / new_done ))
            eta_str="$(format_duration $eta_sec)"
        fi

        local pct=0
        if [ "$batch_total" -gt 0 ]; then
            pct=$(( processed * 100 / batch_total ))
        fi

        cat > "$progress_file" <<EOF
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
shard=$SHARD
total=$batch_total
done=$new_done
failed=$new_fail
elapsed=$(format_duration $elapsed)
eta=$eta_str
pct=${pct}%
EOF

        echo ""
        echo "──────────────────────────────────────────────────"
        echo "[SHARD $SHARD] ${processed}/${batch_total} (${pct}%) | Done: ${new_done} | Failed: ${new_fail} | Elapsed: $(format_duration $elapsed) | ETA: ${eta_str}"
        echo "──────────────────────────────────────────────────"
        echo ""
    done
}

# ========== Signal Handling ==========

monitor_pid=""

cleanup() {
    echo ""
    echo "=========================================="
    echo "Interrupted! Shutting down shard $SHARD..."
    echo "=========================================="

    touch "$stop_file"

    if [ -n "$monitor_pid" ]; then
        kill "$monitor_pid" 2>/dev/null
        wait "$monitor_pid" 2>/dev/null || true
    fi

    local done_count
    done_count=$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
    local fail_count
    fail_count=$(find "$failed_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)

    rm -f "$stop_file"

    echo ""
    echo "Progress saved: $done_count done, $fail_count failed"
    echo "Resume:  RESUME=1 SHARD=$SHARD TOTAL_SHARDS=$TOTAL_SHARDS bash scripts/run_grouping_invariance.sh"
    echo ""
    exit 130
}
trap cleanup SIGINT SIGTERM

# ========== Build Job List (Shard-aware) ==========

total_all=0
skipped=0
jobs=()

job_index=0
for method in "${methods[@]}"; do
    for dataset_entry in "${datasets[@]}"; do
        IFS='|' read -r dataset root_path data_path N ds_e_layers ds_d_ff ds_lr ds_batch ds_data ds_enc_in ds_num_groups <<< "$dataset_entry"
        for pred_len in "${pred_lens[@]}"; do
            for seed in "${seeds[@]}"; do
                total_all=$((total_all + 1))

                # Shard filter: only process jobs assigned to this shard
                if [ $((job_index % TOTAL_SHARDS)) -ne "$SHARD" ]; then
                    job_index=$((job_index + 1))
                    continue
                fi
                job_index=$((job_index + 1))

                job_id="gi_${method}_${dataset}_pl${pred_len}_s${seed}"

                if [ "$RESUME" == "1" ] && [ -f "$done_dir/$job_id" ]; then
                    skipped=$((skipped + 1))
                    continue
                fi

                rm -f "$failed_dir/$job_id" 2>/dev/null || true
                jobs+=("${method}|${dataset}|${root_path}|${data_path}|${N}|${pred_len}|${seed}|${job_id}|${ds_e_layers}|${ds_d_ff}|${ds_lr}|${ds_batch}|${ds_data}|${ds_enc_in}|${ds_num_groups}")
            done
        done
    done
done

remaining=${#jobs[@]}

echo "=========================================="
echo "Grouping Invariance Experiment (Exp 4)"
echo "=========================================="
echo "Shard             : $SHARD / $TOTAL_SHARDS"
echo "Total (all shards): $total_all"
echo "This shard skipped: $skipped"
echo "This shard to run : $remaining"
echo "GPU               : $GPU"
echo "Methods           : ${methods[*]}"
echo "Seeds             : ${seeds[*]} (${#seeds[@]})"
echo "Summary           : $summary_file"
echo "=========================================="
echo ""

if [ "$remaining" -eq 0 ]; then
    echo "All jobs for shard $SHARD already completed!"
    exit 0
fi

# ========== Run Jobs Sequentially ==========

start_time=$(date +%s)
done_count=0
fail_count=0
oom_count=0

initial_done_count=$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
initial_fail_count=$(find "$failed_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)

# Start progress monitor (background)
monitor_progress "$start_time" "$initial_done_count" "$initial_fail_count" "$remaining" &
monitor_pid=$!

echo "[$(date '+%H:%M:%S')] Tip: tail -f $results_dir/<job_id>.log to watch a run"
echo ""

for job in "${jobs[@]}"; do
    # Check for graceful stop
    if [ -f "$stop_file" ]; then
        echo "[$(date '+%H:%M:%S')] Stop requested — exiting."
        break
    fi

    IFS='|' read -r method dataset root_path data_path N pred_len seed job_id ds_e_layers ds_d_ff ds_lr ds_batch ds_data ds_enc_in ds_num_groups <<< "$job"

    # Skip if same config already OOMed on a different seed
    oom_key="${method}_${dataset}_pl${pred_len}"
    if grep -qF "$oom_key" "$oom_skip_file" 2>/dev/null; then
        echo "$method,$dataset,$N,$pred_len,$seed,OOM_SKIP,OOM_SKIP,$job_id" > "$done_dir/$job_id"
        oom_count=$((oom_count + 1))
        echo "[$(date '+%H:%M:%S')] SKIP   ${job_id}  (OOM on same config, skipping seed)"
        continue
    fi

    log_file="$results_dir/${job_id}.log"
    job_start=$(date +%s)

    echo "[$(date '+%H:%M:%S')] START  ${job_id}  ($((done_count + fail_count + 1))/$remaining)"

    if PYTHONUNBUFFERED=1 python run.py \
        --is_training 1 \
        --data "$ds_data" \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$job_id" \
        --model VG_iTransformer \
        --num_groups "$ds_num_groups" \
        --pooling "$pooling" \
        --use_global_interact "$global_interact" \
        "${sf_args[@]}" \
        --custom_grouping_method "$method" \
        --pred_len "$pred_len" \
        --seq_len "$seq_len" \
        --seed "$seed" \
        --e_layers "$ds_e_layers" \
        --d_model "$d_model" \
        --d_ff "$ds_d_ff" \
        --batch_size "$ds_batch" \
        --learning_rate "$ds_lr" \
        --enc_in "$ds_enc_in" \
        --dec_in "$ds_enc_in" \
        --c_out "$ds_enc_in" \
        --output_subdir "grouping_invariance" \
        --itr 1 \
        "${precision_args[@]}" \
        "${compile_args[@]}" \
        "${train_args[@]}" \
        --skip_flops_profiling 1 \
        > "$log_file" 2>&1; then

        # Extract metrics
        mse=$(grep -oP "mse:\K[0-9.]+" "$log_file" 2>/dev/null | tail -1 || true)
        [ -z "$mse" ] && mse="N/A"
        mae=$(grep -oP "mae:\K[0-9.]+" "$log_file" 2>/dev/null | tail -1 || true)
        [ -z "$mae" ] && mae="N/A"

        echo "$method,$dataset,$N,$pred_len,$seed,$mse,$mae,$job_id" > "$done_dir/$job_id"
        done_count=$((done_count + 1))

        elapsed=$(( $(date +%s) - job_start ))
        echo "[$(date '+%H:%M:%S')] DONE   ${job_id} ($(format_duration $elapsed)) MSE=${mse}"
    else
        exit_code=$?
        elapsed=$(( $(date +%s) - job_start ))
        if grep -q "CUDA out of memory\|OutOfMemoryError" "$log_file" 2>/dev/null; then
            echo "$method,$dataset,$N,$pred_len,$seed,OOM,OOM,$job_id" > "$done_dir/$job_id"
            oom_count=$((oom_count + 1))
            echo "[$(date '+%H:%M:%S')] OOM    ${job_id} ($(format_duration $elapsed)) CUDA OOM — marked as done"
            echo "$oom_key" >> "$oom_skip_file"
        elif [ "$exit_code" -eq 137 ] || grep -q "MemoryError\|Cannot allocate memory" "$log_file" 2>/dev/null; then
            echo "$method,$dataset,$N,$pred_len,$seed,RAM_OOM,RAM_OOM,$job_id" > "$done_dir/$job_id"
            oom_count=$((oom_count + 1))
            echo "[$(date '+%H:%M:%S')] OOM    ${job_id} ($(format_duration $elapsed)) RAM OOM (exit=$exit_code) — marked as done"
            echo "$oom_key" >> "$oom_skip_file"
        else
            touch "$failed_dir/$job_id"
            fail_count=$((fail_count + 1))
            echo "[$(date '+%H:%M:%S')] FAILED ${job_id} ($(format_duration $elapsed)) — see $log_file"
        fi
    fi

    # Progress update
    processed=$((done_count + fail_count))
    total_elapsed=$(( $(date +%s) - start_time ))
    if [ $done_count -gt 0 ]; then
        eta_sec=$(( (remaining - processed) * total_elapsed / done_count ))
        echo "  [PROGRESS] ${processed}/${remaining} | Done: ${done_count} | Failed: ${fail_count} | ETA: $(format_duration $eta_sec)"
    fi
    echo ""
done

# ========== Stop Monitor ==========

if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null || true
fi

# ========== Merge Results (this shard) ==========

echo "=========================================="
echo "Merging shard $SHARD results..."
echo "=========================================="

# Merge all available .done files (safe to run from any shard)
echo "method,dataset,N,pred_len,seed,MSE,MAE,model_id" > "$summary_file"
if [ "$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    for f in "$done_dir"/*; do
        [ -f "$f" ] || continue
        cat "$f"
    done | sort -t, -k1,1 -k2,2 -k4,4n -k5,5n >> "$summary_file"
fi
total_done=$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
echo "Results merged to: $summary_file ($total_done rows)"

# ========== Final Report ==========

end_time=$(date +%s)
total_elapsed=$((end_time - start_time))

echo ""
echo "=========================================="
echo "Shard $SHARD Complete!"
echo "=========================================="
echo "This shard done    : $done_count"
echo "This shard failed  : $fail_count"
echo "Wall-clock time    : $(format_duration $total_elapsed)"
echo "=========================================="

if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for f in "$failed_dir"/gi_*; do
        [ -f "$f" ] && echo "  - $(basename "$f")"
    done
    echo ""
    echo "Retry: RESUME=1 SHARD=$SHARD TOTAL_SHARDS=$TOTAL_SHARDS bash scripts/run_grouping_invariance.sh"
    exit 1
fi

echo ""
echo "Shard $SHARD completed successfully!"
