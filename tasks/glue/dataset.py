import evaluate
import numpy as np
import logging
import functools

from tasks.abc_dataset import AbstractDataset
from utils.general import colorstr, colorformat, emojis

task_to_keys = {
    "cola": ("sentence", None),         #   8,551
    "mnli": ("premise", "hypothesis"),  # 392,702
    "mrpc": ("sentence1", "sentence2"), #   3,668
    "qnli": ("question", "sentence"),   # 104,743
    "qqp": ("question1", "question2"),  # 363,846
    "rte": ("sentence1", "sentence2"),  #   2,490
    "sst2": ("sentence", None),         #  67,349
    "stsb": ("sentence1", "sentence2"), #   5,749
    "wnli": ("sentence1", "sentence2"), #     635
}

logger = logging.getLogger(__name__)


class GlueDataset(AbstractDataset):
    def __init__(self, data_args, model_args, training_args, tokenizer):
        super().__init__(data_args, model_args, training_args)
        
        # labels
        self.is_regression = data_args.dataset_name == "stsb"
        if not self.is_regression:
            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1
        
        # Set max_target_length by label_list
        self.set_max_target_length(training_args.generation_max_length)
        
        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
        
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
        
        # Check dataset
        if self.sentence2_key is None:
            logger.info(f"{colorstr('bright_yellow', 'bold', 'Check Dataset')} \n"
                        f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                        f"{colorstr('bright_magenta', 'bold', 'label')} : {self.id2label[self.raw_datasets['train'][0]['label']]}"
                        )
        else:
            logger.info(f"{colorstr('bright_yellow', 'bold', 'Check Dataset')} \n"
                        f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                        f"{colorstr('bright_magenta', 'bold', self.sentence2_key)} : {self.raw_datasets['train'][0][self.sentence2_key]}\n"
                        f"{colorstr('bright_magenta', 'bold', 'label')} : "
                        f"{self.raw_datasets['train'][0]['label'] if self.is_regression else self.id2label[self.raw_datasets['train'][0]['label']]}"  
                        )
        # Preprocess format
        self.processed_dataset = self.preprocess_dataset()
        
        # Tokenize
        self.tokenized_dataset = self.tokenize_dataset()
        
        # Split Dataset
        self.split_dataset()
        
        # Set metrics
        self.set_metrics()
        
    def round_stsb_target(self, label):
        """STSB maps two sentences to a floating point number between 1 and 5
        representing their semantic similarity. Since we are treating all tasks as
        text-to-text tasks we need to convert this floating point number to a string.
        The vast majority of the similarity score labels in STSB are in the set
        [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
        entry in this set, and then we convert the result to a string (literally e.g.
        "3.4"). This converts STSB roughly into a 26-class classification dataset.
        Args:
        label: original label.
        Returns:
        A preprocessed label.
        """
        return np.round((label * 5) / 5, decimals=1)
    
    def preprocessor(self, example, add_prefix=True):
        src_texts = [self.sentence1_key + ": ", example[self.sentence1_key]]
        if self.sentence2_key is not None:
            src_texts.append(self.sentence2_key + ": ")
            src_texts.append(example[self.sentence2_key])
        
        tgt_texts = [str(example['label']) if not self.is_regression 
                     else str(self.round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
    def split_dataset(self):
        # Training
        if training_args.do_train:
            self.train_dataset = self.tokenized_dataset['train']
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        
        # Evaluation
        if training_args.do_eval:
            self.eval_dataset = self.tokenized_dataset['validation_matched' if data_args.dataset_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        
        # Prediction
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = self.tokenized_dataset['test_matched' if data_args.dataset_name == 'mnli' else 'test']
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

    def set_metrics(self):
        if data_args.dataset_name is not None:
            self.metric = evaluate.load("glue", data_args.dataset_name)
        elif self.is_regression:
            self.metric = evaluate.load("mse")
        else:
            self.metric = evaluate.load("accuracy")