#!/bin/bash
# ==========================================================
# Industrial Dataset Benchmark — iT vs iFlashformer vs SF VG-iT
#
# Purpose: Compare iTransformer (baseline), iFlashformer (kernel-level
#   efficiency via FlashAttention), and SF VG-iT (algorithm-level
#   efficiency via shifted grouping + FiLM broadcast) on 6 industrial
#   datasets with high-dimensional variates.
#
# Datasets (N = number of variates):
#   BASF       : N~244,  hourly,   manufacturing process
#   KAMP       : N~228,  minutely, battery charging
#   BDG2       : N~2817, hourly,   building energy (8 meter types)
#   ASHRAE     : N~2362, hourly,   building energy (wide pivot)
#   SDWPF      : N~2144, minutely, wind power (134 turbines)
#   CARE Wind  : N~238,  minutely, single wind turbine sensors
#
# Models:
#   iTransformer  (iT)   : baseline, full N×N attention
#   iFlashformer  (iFlash): FlashAttention kernel, full N×N attention
#   SF VG-iT              : shifted grouping + FiLM broadcast
#
# Total: 6 datasets × 4 PLs × 5 seeds × 3 models = 360 runs
#
# Usage:
#   bash scripts/run_industrial_benchmark.sh
#   GPU=1 bash scripts/run_industrial_benchmark.sh
#   DEBUG=1 bash scripts/run_industrial_benchmark.sh      # 2 epochs
#   RESUME=0 bash scripts/run_industrial_benchmark.sh     # no resume
#
# Graceful stop:
#   touch ./test_results/industrial_benchmark/.stop
# ==========================================================

set -eo pipefail

# ========== Configuration ==========

GPU="${GPU:-0}"
RESUME="${RESUME:-1}"

if [ "${DEBUG:-0}" == "1" ]; then
    echo "!!! DEBUG MODE: 2 epochs !!!"
    train_args=(--train_epochs 2 --patience 1 --num_workers 0)
else
    train_args=(--train_epochs 100 --patience 5 --num_workers 4)
fi

export CUDA_VISIBLE_DEVICES=$GPU

# Fixed parameters
pooling=mean
seq_len=96
d_model=512

# Precision (bf16)
precision_args=(--use_amp --amp_dtype bf16)
compile_args=(--use_compile)

# SF flags
sf_args=(--use_shifted_grouping 1 --use_film_broadcast 1)

# Datasets: "name|root_path|data_path|freq|batch_base|batch_vgit|num_groups|e_layers|d_ff|lr"
# enc_in is auto-detected from CSV header
# batch_base: batch size for iT / iFlashformer (smaller for high-N to avoid OOM)
# batch_vgit: batch size for VG-iT
datasets=(
    "basf|./dataset/basf/|basf.csv|h|16|16|16|3|512|0.0005"
    "kamp|./dataset/kamp/|kamp.csv|t|16|16|16|3|512|0.0005"
    "bdg2|./dataset/bdg2/|bdg2.csv|h|8|8|64|3|512|0.0005"
    "ashrae|./dataset/ashrae_energy/|ashrae.csv|h|8|8|64|3|512|0.0005"
    "sdwpf|./dataset/sdwpf_energy/|sdwpf.csv|t|8|8|64|3|512|0.0005"
    "care_wind|./dataset/care_wind_energy/|care_wind.csv|t|16|16|16|3|512|0.0005"
)

pred_lens=(96 192 336 720)
seeds=(2021 2022 2023 2024 2025)

# Directories
results_dir="./test_results/industrial_benchmark"
done_dir="$results_dir/.done"
failed_dir="$results_dir/.failed"
stop_file="$results_dir/.stop"
summary_file="./test_results/industrial_benchmark_results.csv"

oom_skip_file="$results_dir/.oom_skip"

mkdir -p "$results_dir" "$done_dir" "$failed_dir"
rm -f "$stop_file"

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

get_enc_in() {
    # Auto-detect number of variates from CSV header
    # Counts columns minus 'date' column
    local csv_path="$1"
    if [ ! -f "$csv_path" ]; then
        echo "ERROR: $csv_path not found" >&2
        return 1
    fi
    local n_cols
    n_cols=$(head -1 "$csv_path" | tr ',' '\n' | wc -l)
    echo $((n_cols - 1))  # subtract 'date' column
}

# ========== Build Job List ==========

total_all=0
skipped=0
jobs=()

for dataset_entry in "${datasets[@]}"; do
    IFS='|' read -r dataset root_path data_path freq batch_base batch_vgit num_groups ds_e_layers ds_d_ff ds_lr <<< "$dataset_entry"

    # Auto-detect enc_in
    csv_file="${root_path}${data_path}"
    if [ ! -f "$csv_file" ]; then
        echo "[WARNING] Dataset not found: $csv_file — skipping $dataset"
        echo "  Run preprocessing first: python scripts/preprocess_industrial/preprocess_${dataset}.py"
        continue
    fi
    enc_in=$(get_enc_in "$csv_file")
    echo "[INFO] $dataset: enc_in=$enc_in (from $csv_file)"

    for seed in "${seeds[@]}"; do
        for pred_len in "${pred_lens[@]}"; do

            # --- iTransformer baseline ---
            total_all=$((total_all + 1))
            job_id="ind_iT_${dataset}_pl${pred_len}_s${seed}"

            if [ "$RESUME" == "1" ] && [ -f "$done_dir/$job_id" ]; then
                skipped=$((skipped + 1))
            else
                rm -f "$failed_dir/$job_id" 2>/dev/null || true
                jobs+=("iT|${dataset}|${root_path}|${data_path}|${enc_in}|${pred_len}|${seed}|${job_id}|${ds_e_layers}|${ds_d_ff}|${ds_lr}|${batch_base}|${freq}|${num_groups}")
            fi

            # --- iFlashformer ---
            total_all=$((total_all + 1))
            job_id="ind_iFlash_${dataset}_pl${pred_len}_s${seed}"

            if [ "$RESUME" == "1" ] && [ -f "$done_dir/$job_id" ]; then
                skipped=$((skipped + 1))
            else
                rm -f "$failed_dir/$job_id" 2>/dev/null || true
                jobs+=("iFlash|${dataset}|${root_path}|${data_path}|${enc_in}|${pred_len}|${seed}|${job_id}|${ds_e_layers}|${ds_d_ff}|${ds_lr}|${batch_base}|${freq}|${num_groups}")
            fi

            # --- SF VG-iT ---
            total_all=$((total_all + 1))
            job_id="ind_sfvgit_${dataset}_pl${pred_len}_s${seed}"

            if [ "$RESUME" == "1" ] && [ -f "$done_dir/$job_id" ]; then
                skipped=$((skipped + 1))
            else
                rm -f "$failed_dir/$job_id" 2>/dev/null || true
                jobs+=("SF_VGiT|${dataset}|${root_path}|${data_path}|${enc_in}|${pred_len}|${seed}|${job_id}|${ds_e_layers}|${ds_d_ff}|${ds_lr}|${batch_vgit}|${freq}|${num_groups}")
            fi

        done
    done
done

remaining=${#jobs[@]}

echo ""
echo "=========================================="
echo "Industrial Dataset Benchmark"
echo "=========================================="
echo "Total experiments : $total_all"
echo "Already completed : $skipped"
echo "To run            : $remaining"
echo "GPU               : $GPU"
echo "Models            : iT, iFlashformer, SF VG-iT"
echo "Pred lengths      : ${pred_lens[*]}"
echo "Seeds             : ${seeds[*]}"
echo "Summary           : $summary_file"
echo "=========================================="
echo ""

if [ "$remaining" -eq 0 ]; then
    echo "All experiments already completed!"
    # Merge results
    echo "model,dataset,N,pred_len,seed,MSE,MAE,freq,model_id" > "$summary_file"
    if [ "$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        for f in "$done_dir"/*; do
            [ -f "$f" ] || continue
            cat "$f"
        done | sort -t, -k2,2 -k1,1 -k4,4n >> "$summary_file"
    fi
    echo "Results merged to: $summary_file"
    exit 0
fi

# ========== Run Jobs Sequentially ==========

start_time=$(date +%s)
done_count=0
fail_count=0
oom_count=0

for job in "${jobs[@]}"; do
    # Check for graceful stop
    if [ -f "$stop_file" ]; then
        echo "[$(date '+%H:%M:%S')] Stop requested — exiting."
        break
    fi

    IFS='|' read -r model dataset root_path data_path enc_in pred_len seed job_id ds_e_layers ds_d_ff ds_lr batch_size freq num_groups <<< "$job"

    # Skip if same config already OOMed on a different seed
    oom_key="${model}_${dataset}_pl${pred_len}"
    if grep -qF "$oom_key" "$oom_skip_file" 2>/dev/null; then
        echo "$model,$dataset,$enc_in,$pred_len,$seed,OOM_SKIP,OOM_SKIP,$freq,$job_id" > "$done_dir/$job_id"
        oom_count=$((oom_count + 1))
        echo "[$(date '+%H:%M:%S')] SKIP   ${job_id}  (OOM on same config, skipping seed)"
        continue
    fi

    log_file="$results_dir/${job_id}.log"
    job_start=$(date +%s)

    echo "[$(date '+%H:%M:%S')] START  ${job_id}  ($((done_count + fail_count + 1))/$remaining)"

    # Build model-specific args
    if [ "$model" == "iT" ]; then
        model_name="iTransformer"
        model_args=()
    elif [ "$model" == "iFlash" ]; then
        model_name="iFlashformer"
        model_args=()
    else
        model_name="VG_iTransformer"
        model_args=(
            --num_groups "$num_groups"
            --pooling "$pooling"
            --use_global_interact 1
            "${sf_args[@]}"
        )
    fi

    if PYTHONUNBUFFERED=1 python run.py \
        --is_training 1 \
        --data custom \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$job_id" \
        --model "$model_name" \
        "${model_args[@]}" \
        --features M \
        --freq "$freq" \
        --pred_len "$pred_len" \
        --seq_len "$seq_len" \
        --seed "$seed" \
        --e_layers "$ds_e_layers" \
        --d_model "$d_model" \
        --d_ff "$ds_d_ff" \
        --batch_size "$batch_size" \
        --learning_rate "$ds_lr" \
        --enc_in "$enc_in" \
        --dec_in "$enc_in" \
        --c_out "$enc_in" \
        --output_subdir "industrial_benchmark" \
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

        echo "$model,$dataset,$enc_in,$pred_len,$seed,$mse,$mae,$freq,$job_id" > "$done_dir/$job_id"
        done_count=$((done_count + 1))

        elapsed=$(( $(date +%s) - job_start ))
        echo "[$(date '+%H:%M:%S')] DONE   ${job_id} ($(format_duration $elapsed)) MSE=${mse}"
    else
        exit_code=$?
        elapsed=$(( $(date +%s) - job_start ))
        if grep -q "CUDA out of memory\|OutOfMemoryError" "$log_file" 2>/dev/null; then
            echo "$model,$dataset,$enc_in,$pred_len,$seed,OOM,OOM,$freq,$job_id" > "$done_dir/$job_id"
            oom_count=$((oom_count + 1))
            echo "[$(date '+%H:%M:%S')] OOM    ${job_id} ($(format_duration $elapsed)) CUDA OOM — marked as done"
            echo "$oom_key" >> "$oom_skip_file"
        elif [ "$exit_code" -eq 137 ] || grep -q "MemoryError\|Cannot allocate memory" "$log_file" 2>/dev/null; then
            echo "$model,$dataset,$enc_in,$pred_len,$seed,RAM_OOM,RAM_OOM,$freq,$job_id" > "$done_dir/$job_id"
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
    processed=$((done_count + fail_count + oom_count))
    total_elapsed=$(( $(date +%s) - start_time ))
    if [ $processed -gt 0 ]; then
        eta_sec=$(( (remaining - processed) * total_elapsed / processed ))
        echo "  [PROGRESS] ${processed}/${remaining} | Done: ${done_count} | Failed: ${fail_count} | ETA: $(format_duration $eta_sec)"
    fi
    echo ""
done

# ========== Merge Results ==========

echo "=========================================="
echo "Merging results..."
echo "=========================================="

echo "model,dataset,N,pred_len,seed,MSE,MAE,freq,model_id" > "$summary_file"

if [ "$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    for f in "$done_dir"/*; do
        [ -f "$f" ] || continue
        cat "$f"
    done | sort -t, -k2,2 -k1,1 -k4,4n >> "$summary_file"
fi

# ========== Final Report ==========

end_time=$(date +%s)
total_elapsed=$((end_time - start_time))
done_count_final=$(find "$done_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
fail_count_final=$(find "$failed_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "Industrial Benchmark Complete!"
echo "=========================================="
echo "Total experiments : $total_all"
echo "Completed         : $done_count_final"
echo "Failed            : $fail_count_final"
echo "Wall-clock time   : $(format_duration $total_elapsed)"
echo "Summary CSV       : $summary_file"
echo "=========================================="

if [ "$fail_count_final" -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for f in "$failed_dir"/*; do
        [ -f "$f" ] && echo "  - $(basename "$f")"
    done
    echo ""
    echo "Retry: RESUME=1 bash scripts/run_industrial_benchmark.sh"
    exit 1
fi

echo ""
echo "All experiments completed successfully!"
