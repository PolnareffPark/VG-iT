#!/bin/bash

# ==========================================================
# [Utils] Global Progress Tracking & Robust Execution
# ==========================================================

PROGRESS_FILE="master_progress.cnt"
TOTAL_RUNS=243 # Default, can be overridden

# Default Python Executable
if [ -z "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python"
fi

init_progress() {
    local total=$1
    if [ ! -z "$RESUME_FROM" ]; then
        echo "Resuming from experiment #$RESUME_FROM"
        # We start counting from 0, run_python will increment it.
        # If we want to start AT RESUME_FROM, we set counter to RESUME_FROM - 1.
        echo $((RESUME_FROM - 1)) > $PROGRESS_FILE
    else
        echo "0" > $PROGRESS_FILE
    fi
    # Use the passed total or default
    if [ ! -z "$total" ]; then
        TOTAL_RUNS=$total
    fi
    echo "[Progress Utils] Initialized. Total Runs: $TOTAL_RUNS"
}

model_id_exists_in_csv() {
    local csv_file="$1"
    local target_id="$2"
    awk -F',' -v target="$target_id" '
        NR==1 {
            col=0
            for (i=1; i<=NF; i++) {
                gsub(/^[[:space:]\"]+|[[:space:]\"]+$/, "", $i)
                if ($i == "model_id") {
                    col=i
                    break
                }
            }
            next
        }
        {
            if (col > 0) {
                v=$col
                gsub(/^[[:space:]\"]+|[[:space:]\"]+$/, "", v)
                if (v == target) {
                    found=1
                    exit
                }
            }
        }
        END { exit(found ? 0 : 1) }
    ' "$csv_file"
}

run_python() {
    # 1. Parse Args for Smart Resume (target model_id)
    local target_id=""
    local target_summary=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [ "${args[i]}" == "--model_id" ]; then
            target_id="${args[i+1]}"
        elif [ "${args[i]}" == "--summary_file" ]; then
            target_summary="${args[i+1]}"
        fi
    done

    # 2. Update Global Counter
    if [ ! -f $PROGRESS_FILE ]; then
        echo "0" > $PROGRESS_FILE
    fi
    
    # Read current count safely
    local current=$(cat $PROGRESS_FILE)
    current=$((current + 1))
    echo "$current" > $PROGRESS_FILE
    
    # 3. Smart Resume: Skip if model_id already exists in result files
    if [ ! -z "$target_id" ]; then
        # Check all relevant summary files
        local check_files=("test_results/summary.csv" "test_results/W1_results.csv" "test_results/A1_results.csv" "test_results/A2_results.csv" "test_results/A3_results.csv")
        if [ ! -z "$target_summary" ]; then
            check_files+=("test_results/$target_summary")
        fi
        for f in "${check_files[@]}"; do
            if [ -f "$f" ]; then
                if model_id_exists_in_csv "$f" "$target_id"; then
                    echo ">> [Global: ${current}/${TOTAL_RUNS}] SKIPPING: '$target_id' already completed (found in $f)"
                    return 0
                fi
            fi
        done
    fi

    # 4. Fallback: Sequential Resume logic (if RESUME_FROM is set)
    if [ ! -z "$RESUME_FROM" ]; then
        if [ "$current" -lt "$RESUME_FROM" ]; then
            echo ">> [Global: ${current}/${TOTAL_RUNS}] SKIPPING (Sequential Resume, target: $RESUME_FROM)"
            return 0
        fi
    fi

    # 5. Log Progress
    echo "----------------------------------------------------------"
    echo ">> [Global: ${current}/${TOTAL_RUNS}] Executing..."
    echo ">> Cmd: $PYTHON_EXEC -u run.py $@"
    echo "----------------------------------------------------------"
    
    # 6. Execute with Error Handling
    $PYTHON_EXEC -u run.py "$@"
    local status=$?
    
    if [ $status -ne 0 ]; then
        echo "!!!! [Global: ${current}/${TOTAL_RUNS}] FAILED (Exit Code: $status) !!!!"
    else
        echo ">> [Global: ${current}/${TOTAL_RUNS}] COMPLETED Successfully."
    fi
}
