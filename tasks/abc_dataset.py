import abc
from abc import abstractmethod
import logging
import numpy as np

from typing import Callable, List, Mapping
from transformers import (
    AutoTokenizer,
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction
)
from datasets import load_dataset

logger = logging.getLogger(__name__)


class AbstractDataset(abc.ABC):
    task = NotImplemented
    name = NotImplemented
    data_args = NotImplemented
    training_args = NotImplemented
    model_args = NotImplemented
    tokenizer = NotImplemented
    
    raw_datasets = NotImplemented
    processed_dataset = NotImplemented
    tokenized_dataset = NotImplemented
    preprocessor: Callable = NotImplemented
    prefix = NotImplemented
    metric = NotImplemented
    
    train_dataset = NotImplemented
    eval_dataset = NotImplemented
    predict_dataset = NotImplemented
    
    labels_list = None
    num_labels = None
    label2id = None
    id2label = None
    
    max_target_length = NotImplemented
    max_seq_length = NotImplemented
    data_collator = NotImplemented
    column_names = ['source', 'target']
    
    def __init__(self, data_args, model_args, training_args, tokenizer):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        
        self.task = data_args.task_name
        self.name = data_args.dataset_name
        
        if self.task in ['glue', 'super_glue']:
            self.raw_datasets = load_dataset(self.task, self.name)
        else:
            raise NotImplementedError

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = 'max_length'
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False
        
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger then the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            self.data_collator = None
    
    def set_max_target_length(self, default_max_length):
        if self.labels_list is not None:
            self.max_target_length = max([len(self.tokenizer.encode(label)) for label in self.labels_list])
        self.max_target_length = default_max_length
    
    def seq2seq_format(self,
                       sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix] + sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                'task': self.name}
    
    def preprocess_dataset(self):
        return self.raw_datasets.map(
            functools.partial(self.preprocessor, add_prefix=self.data_args.add_prefix),
            remove_columns=self.raw_datasets['train'].column_names
        )
    
    def tokenize_dataset(self):
        if any(x in self.model_args.model_name_or_path for x in ["bert", "roberta", "albert"]):
            logger.info(f"Loading encoder model from {self.model_args.model_name_or_path}")
            tokenize_function = self.encoder_preprocess_function
            self.compute_metrics = self.compute_metrics_encoder
        elif any(x in self.model_args.model_name_or_path for x in ["t5"]):
            logger.info(f"Loading seq2seq model from {self.model_args.model_name_or_path}")
            tokenize_function = self.seq2seq_preprocess_function
            self.compute_metrics = self.compute_metrics_seq2seq
        elif any(x in self.model_args.model_name_or_path for x in ["gpt", "llama"]):
            logger.info(f"Loading decoder model from {self.model_args.model_name_or_path}")
            tokenize_function = self.decoder_preprocess_function
            self.compute_metrics = self.compute_metrics_decoder
        else:
            raise NotImplementedError
        
        return self.processed_dataset.map(
            functools.partial(tokenize_function),
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
        )
    
    @abstractmethod
    def split_dataset(self):
        # Split Dataset train, evaluate, pedict
        pass
    
    @abstractmethod
    def set_metrics(self):
        pass
    
    # Preprocessing tokenize datasets
    def encoder_preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples['source'],
                                      max_length=self.max_seq_length,
                                      padding=self.padding,
                                      truncation=True)
        # Setup the tokenizer for targets
        labels = torch.tensor([int(i) for i in examples['target']])
        model_inputs['labels'] = labels
        return model_inputs
    
    def seq2seq_preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples['source'],
                                      max_length=self.max_seq_length,
                                      padding=self.padding,
                                      truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['target'],
                                    max_length=self.max_target_length,
                                    padding=self.padding,
                                    truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == 'max_length' and self.data_args.ignore_pad_token_for_loss:
            labels['input_ids'] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']
            ]
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def decoder_preprocess_function(self, examples):
        batch_size = len(examples['source'])
        inputs = [f"{x} Label : " for x in examples['source']]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(examples['target'])
        
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.data_args.max_seq_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.data_args.max_seq_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.data_args.max_seq_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.data_args.max_seq_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.data_args.max_seq_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.data_args.max_seq_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # PostProcessing
    def postprocess(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can'y decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels
    
    def compute_metrics_encoder(self, p: EvalPrediction):
        preds, labels = p
        num_logits = preds.shape[-1]
        if num_logits == 1:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        result = self.metric.compute(predictions=preds, references=labels)
        return result
    
    def compute_metrics_seq2seq(self, p: EvalPrediction):
        preds, labels = p
        decoded_preds, decoded_labels = self.postprocess(preds, labels, data_info=None)
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result
    
    def compute_metrics_decoder(self, p: EvalPrediction):
        output_sequences, labels = p
        preds = output_sequences[:, self.data_args.max_seq_length:]
        decoded_preds, decoded_labels = self.postprocess(preds, labels, data_info=None)
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result