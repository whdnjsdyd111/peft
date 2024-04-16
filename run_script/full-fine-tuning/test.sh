for DATASET_NAME in boolq cb rte wic wsc copa record multirc; do
  if test "$DATASET_NAME" = "multirc"; then max_seq_length=348; else max_seq_length=256; fi
  if [ "$DATASET_NAME" = "boolq" ] || [ "$DATASET_NAME" = "cb" ]; then
    epochs=20  # Set 20 epochs for boolq and cb
  elif [ "$DATASET_NAME" = "multirc" ]; then
    epochs=10  # Set 10 epochs for mrpc
  fi
  echo $max_seq_length
  echo $epochs
done;