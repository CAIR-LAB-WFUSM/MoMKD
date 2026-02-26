#!/usr/bin/env bash


TASK_NAME="her2"      
MODEL_NAME="mkd"     

WSI_FEATURE_DIR="..."
RNA_FEATURE_DIR="..."

if [ "$TASK_NAME" == "odx" ]; then
    echo "--- CONFIGURING FOR HER2 STATUS TASK ---"
    SPLIT_DIR_NAME="..."
    MAIN_CSV_PATH="..."
    LABEL_COLUMN="label"
    POSITIVE_LABEL_VALUE="..."
    RESULTS_PARENT_DIR_BASE="..." 
    mkdir -p "$RESULTS_PARENT_DIR_BASE"
fi 

RNA_FEATURE_DIM=768
SEED=4
K_FOLDS=1
MAX_EPOCHS=100
GRAD_ACCUMULATION=16

EARLY_STOPPING_ENABLED=true
PATIENCE=1
STOP_EPOCH=1

LEARNING_RATES=(2e-4)
memory_size=(16)

WSI_ENCODER_DIM=512
RNA_EMBEDDING_DIM=512
SHARED_EMBEDDING_DIM=64
CLS_HIDDEN_DIM=128

EARLY_STOPPING_FLAG=""
if [ "$EARLY_STOPPING_ENABLED" = true ]; then
    EARLY_STOPPING_FLAG="--early_stopping --patience $PATIENCE --stop_epoch $STOP_EPOCH"
    echo "Early stopping enabled with patience=$PATIENCE, starting at epoch=$STOP_EPOCH."
fi

for LR in "${LEARNING_RATES[@]}"; do
    for num_memory in "${memory_size[@]}"; do

        GRID_RUN_ID="lr_${LR}_proto_${num_memory}"
        RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR_BASE}/${GRID_RUN_ID}"
      

        echo "--- PHASE 1: TRAINING ---"
     
        python main.py \
            --model_name "$MODEL_NAME" \
            --task_type "$TASK_NAME" \
            --mode coattn \
            --data_root_dir "$WSI_FEATURE_DIR" \
            --omic_root_dir "$RNA_FEATURE_DIR" \
            --csv_path "$MAIN_CSV_PATH" \
            --label_col "$LABEL_COLUMN" \
            --positive_label "$POSITIVE_LABEL_VALUE" \
            --split_dir "$SPLIT_DIR_NAME" \
            --results_dir "$RESULTS_PARENT_DIR" \
            --lr "$LR" \
            --max_epochs $MAX_EPOCHS \
            --gc $GRAD_ACCUMULATION \
            --seed $SEED \
            --k $K_FOLDS \
            --num_memory $num_memory \
            $EARLY_STOPPING_FLAG \
            || { echo "[ERROR] Training script failed."; exit 1; }

        echo "--- PHASE 2: TESTING ---"
        python test.py \
            --model_name "$MODEL_NAME" \
            --omic_input_dim $RNA_FEATURE_DIM \
            --results_dir "$RESULTS_PARENT_DIR" \
            --split_dir_name "$SPLIT_DIR_NAME" \
            --data_root_dir "$WSI_FEATURE_DIR" \
            --csv_path "$MAIN_CSV_PATH" \
            --label_col "$LABEL_COLUMN" \
            --positive_label "$POSITIVE_LABEL_VALUE" \
            --seed $SEED \
            --k $K_FOLDS \
            --num_memory $num_memory

    done 
done