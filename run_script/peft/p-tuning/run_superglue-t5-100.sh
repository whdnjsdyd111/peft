export MODELS_NAME="t5-base t5-large"
export TASK_NAME=super_glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=P_TUNING

max_seq_length=256
bs=8
epochs=20
seeds="40 41 42"
lrs="0.01 0.1 0.3 0.7 1.0"
weight_decay=0.01
virtual_tokens_list="100"

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in boolq cb rte wic wsc copa record multirc; do
    for lr in $lrs; do
      for seed in $seeds; do
        for var in $virtual_tokens_list; do
          if test "$DATASET_NAME" = "multirc"; then max_seq_length=348; fi
          python run.py \
            --model_name_or_path $MODEL_NAME \
            --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$seed-$PEFT_TYPE-$var-token \
            --task_name $TASK_NAME \
            --dataset_name $DATASET_NAME \
            --do_train \
            --do_eval \
            --do_predict \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --max_seq_length $max_seq_length \
            --output_dir checkpoints/PEFT/$PEFT_TYPE/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$seed-$var-token/ \
            --overwrite_output_dir \
            --seed $seed \
            --learning_rate $lr \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --num_train_epochs $epochs \
            --weight_decay $weight_decay \
            --load_best_model_at_end \
            --save_total_limit 1 \
            --peft_type $PEFT_TYPE \
            --num_virtual_tokens $var;
        done;
      done;
    done;
  done;
done;