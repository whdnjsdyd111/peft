export MODELS_NAME="bert-base-uncased bert-large-uncased"
export TASK_NAME=glue
export CUDA_VISIBLE_DEVICES=0
export PEFT_TYPE=PREFIX_TUNING

seed=42
lrs="1e-6 5e-5 1e-5 5e-4 1e-4"
max_seq_length=256
bs=32
max_steps=30000
weight_decay=1e-5
virtual_tokens_list="10"

for MODEL_NAME in $MODELS_NAME; do
  for DATASET_NAME in cola mrpc rte stsb wnli mnli qnli qqp sst2; do
    for lr in $lrs; do
      for var in $virtual_tokens_list; do
        python run.py \
          --model_name_or_path $MODEL_NAME \
          --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-$PEFT_TYPE-$var-token \
          --task_name $TASK_NAME \
          --dataset_name $DATASET_NAME \
          --do_train \
          --do_eval \
          --do_predict \
          --per_device_train_batch_size $bs \
          --per_device_eval_batch_size $bs \
          --max_seq_length $max_seq_length \
          --output_dir checkpoints/PEFT/$PEFT_TYPE/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-$var-token/ \
          --overwrite_output_dir \
          --seed $seed \
          --learning_rate $lr \
          --save_strategy steps \
          --evaluation_strategy steps \
          --max_steps $max_steps \
          --eval_steps 1000 \
          --save_steps 1000 \
          --warmup_steps 500 \
          --weight_decay $weight_decay \
          --load_best_model_at_end \
          --save_total_limit 1 \
          --peft_type $PEFT_TYPE \
          --num_virtual_tokens $var;
      done;
    done;
  done;
done;