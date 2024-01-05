export MODEL_NAME=bert-base-uncased
export TASK_NAME=glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=PREFIX_TUNING

max_seq_length=256
bs=32
epoch=20
weight_decay=1e-5
virtual_tokens_list="20"

for DATASET_NAME in cola mrpc rte stsb wnli; do
  for lr in 5e-4 5e-5; do
    for var in $virtual_tokens_list; do
      python run.py \
        --model_name_or_path $MODEL_NAME \
        --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$PEFT_TYPE-$var-k-shot-10 \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --max_seq_length $max_seq_length \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --weight_decay $weight_decay \
        --output_dir checkpoints/PEFT/PREFIX_TUNING/$TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$PEFT_TYPE-$var-k-shot-10/ \
        --overwrite_output_dir \
        --seed 1 \
        --save_strategy no \
        --evaluation_strategy epoch \
        --peft_type $PEFT_TYPE \
        --k_shot_example 10 \
        --num_virtual_tokens $var
    done;
  done;
done