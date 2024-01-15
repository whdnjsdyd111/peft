export MODEL_NAME=t5-base
export TASK_NAME=glue
export CUDA_VISIBLE_DEVICES=0

max_seq_length=256
bs=32
max_steps=30000
weight_decay=1e-5

for DATASET_NAME in cola mrpc rte stsb wnli mnli qnli qqp sst2; do
  for lr in 1e-4 5e-4 1e-3; do
    python run.py \
      --model_name_or_path $MODEL_NAME \
      --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr-10-shot \
      --task_name $TASK_NAME \
      --dataset_name $DATASET_NAME \
      --do_train \
      --do_eval \
      --do_predict \
      --per_device_train_batch_size $bs \
      --per_device_eval_batch_size $bs \
      --max_seq_length $max_seq_length \
      --output_dir checkpoints/FFT/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr-10-shot/ \
      --overwrite_output_dir \
      --seed 1 \
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
      --k_shot_example 10;
  done;
done