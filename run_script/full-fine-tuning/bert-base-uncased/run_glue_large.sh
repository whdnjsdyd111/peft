export MODEL_NAME=bert-base-uncased
export TASK_NAME=glue
export CUDA_VISIBLE_DEVICES=0

max_seq_length=256
bs=32
epoch=10
weight_decay=1e-5

for DATASET_NAME in mnli qnli qqp sst2; do
  for lr in 1e-4 5e-4 1e-3; do
    python run.py \
      --model_name_or_path $MODEL_NAME \
      --run_name $TASK_NAME-$DATASET_NAME-$MODEL_NAME-$lr \
      --task_name $TASK_NAME \
      --dataset_name $DATASET_NAME \
      --do_train \
      --do_eval \
      --do_predict \
      --max_seq_length $max_seq_length \
      --per_device_train_batch_size $bs \
      --per_device_eval_batch_size $bs \
      --learning_rate $lr \
      --num_train_epochs $epoch \
      --weight_decay $weight_decay \
      --output_dir checkpoints/FFT/$MODEL_NAME/$TASK_NAME-$DATASET_NAME-$lr/ \
      --overwrite_output_dir \
      --seed 1 \
      --save_strategy no \
      --evaluation_strategy epoch \
      --save_strategy epoch \
      --load_best_model_at_end \
      --save_total_limit 1 
  done;
done