import re
import evaluate
import numpy as np
import logging
import functools

from tasks.abc_dataset import AbstractDataset
from utils.general import colorstr, colorformat, emojis

task_to_keys = {
    "boolq": ("question", "passage"),               #   9,427
    "cb": ("premise", "hypothesis"),                #     250
    "rte": ("premise", "hypothesis"),               #   2,490
    "wic": ("sentence1", "sentence2"),              #   5,428
    "wsc": ("span1_text", "span2_text"),            #     554
    "copa": (None, None),                           #     400
    "record": (None, None),                         # 100,730
    "multirc": ("paragraph", "question_answer")     #  27,243
}

logger = logging.getLogger(__name__)


class SuperGlueDataset(AbstractDataset):
    def __init__(self, data_args, model_args, training_args, tokenizer):
        super().__init__(data_args, model_args, training_args, tokenizer)
        
        # labels
        if data_args.dataset_name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        else:
            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        
        # Set max_target_length by label_list
        self.set_max_target_length(training_args.generation_max_length)
        
        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
        
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}
        
        # Check dataset
        logger.info(f"{colorstr('bright_yellow', 'bold', 'Check Dataset')}")
        if self.name == "wic":
            print(f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', self.sentence2_key)} : {self.raw_datasets['train'][0][self.sentence2_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', 'word')} : {self.raw_datasets['train'][0]['word']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : "
                  f"{self.id2label[self.raw_datasets['train'][0]['label']]}"
                  )
        elif self.name == "wsc":
            print(f"{colorstr('bright_magenta', 'bold', 'text')} : {self.raw_datasets['train'][0]['text']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'span')} : {self.raw_datasets['train'][0][self.sentence2_key]}, {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : "
                  f"{self.id2label[self.raw_datasets['train'][0]['label']]}"
                  )
        elif self.name == "copa":
            print(f"{colorstr('bright_magenta', 'bold', 'premise')} : {self.raw_datasets['train'][0]['premise']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'choice1')} : {self.raw_datasets['train'][0]['choice1']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'choice2')} : {self.raw_datasets['train'][0]['choice2']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : "
                  f"{self.id2label[self.raw_datasets['train'][0]['label']]}"
                  )
        elif self.name == "multirc":
            print(f"{colorstr('bright_magenta', 'bold', 'question')} : {self.raw_datasets['train'][0]['question']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'answer')} : {self.raw_datasets['train'][0]['answer']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'paragraph')} : {self.raw_datasets['train'][0]['paragraph']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : "
                  f"{self.id2label[self.raw_datasets['train'][0]['label']]}"
                  )
        elif self.name == "record":
            print(f"{colorstr('bright_magenta', 'bold', 'passage')} : {self.raw_datasets['train'][0]['passage']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'query')} : {self.raw_datasets['train'][0]['query']}\n"
                  f"{colorstr('bright_magenta', 'bold', 'entities')} : {str(self.raw_datasets['train'][0]['entities'])}\n"
                  f"{colorstr('bright_magenta', 'bold', 'answers')} : {str(self.raw_datasets['train'][0]['answers'])}"
                  )
        else:
            print(f"{colorstr('bright_magenta', 'bold', self.sentence1_key)} : {self.raw_datasets['train'][0][self.sentence1_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', self.sentence2_key)} : {self.raw_datasets['train'][0][self.sentence2_key]}\n"
                  f"{colorstr('bright_magenta', 'bold', 'label')} : "
                  f"{self.id2label[self.raw_datasets['train'][0]['label']]}"
                  )
        
        # Preprocess format
        if data_args.dataset_name == "record":
            self.processed_dataset = self.raw_datasets.map(
                functools.partial(
                    self.preprocessor_record_function, add_prefix=self.data_args.add_prefix),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=self.raw_datasets['train'].column_names
            )
        else:
            self.preprocess_dataset()
        
        if data_args.k_shot_example is not None:
            self.preprocess_k_shot_dataset()
        
        # Tokenize
        self.tokenize_dataset()
        
        # Split Dataset
        self.split_dataset()
        
        # Set metrics
        self.set_metrics()

    def preprocessor(self, example, add_prefix=True):
        extra_fields = {}
        
        if self.name == "wic":
            if self.data_args.template_id == 0: # Seq2Seq Format
                src_texts = [self.sentence1_key + ":", example[self.sentence1_key],
                            self.sentence2_key + ":", example[self.sentence2_key],
                            "word:", example["word"]]
            elif self.data_args.template_id == 1: # ROBERTA
                src_texts = [
                    f"{example[self.sentence1_key]} {example[self.sentence2_key]} Does {example['word']} have the same meaning in both sentences?"
                ]
            elif self.data_args.template_id == 2: # BERT
                src_texts = [example['word'] + ":", example[self.sentence1_key],
                             example['word'] + ":", example[self.sentence2_key]]
        
        elif self.name == "wsc":
            if self.data_args.template_id == 0:
                
                def _mark_span(text, span_str, span_idx, mark):
                    pattern_tmpl = r'^((?:\S+\s){N})(W)'
                    pattern = re.sub('N', str(span_idx), pattern_tmpl)
                    pattern = re.sub('W', span_str, pattern)
                    return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)
                
                text = example['text']
                text = _mark_span(text, example[self.sentence1_key], example['span1_index'], "*")
                # Compensate for 2 added "words" added in previous step.
                span2_index = example['span2_index'] + 2 * \
                    int(example['span1_index'] < example['span2_index'])
                text = _mark_span(text, example[self.sentence2_key], span2_index, "#")
                
                src_texts = ["text:", text]
            elif self.data_args.template_id == 1:
                # TODO
                src_texts = [
                    example[self.sentence2_key] + ":" + example["text"], self.tokenizer.sep_token,
                    example[self.sentence1_key]
                ]
        
        elif self.name == "copa":
            if self.data_args.template_id == 0:
                src_texts = ["premise:", example["premise"],
                             "choice1:", example["choice1"],
                             "choice2:", example["choice2"]
                             ]
            elif self.data_args.template_id == 1:
                joiner = "because" if example["question"] == "cause" else "so"
                text = f"{example['premise']} {joiner}"
                
                src_texts = [text, self.tokenizer.sep_token, example["choice1"], self.tokenizer.eos_token,
                             text, self.tokenizer.sep_token, example["choice2"]
                             ]
        
        elif self.name == "multirc":
                
            def _remove_markup(text):
                """ Removes the HTML markup. """
                text = re.sub('<br>', ' ', text)
                text = re.sub('<(/)?b>', '', text)
                return text

            if self.data_args.template_id == 0:
                group = example['idx']['question']
                
                src_texts = ["question:", _remove_markup(example["question"]),
                             "answer:", _remove_markup(example["answer"]),
                             "paragraph", _remove_markup(example["paragraph"])]
                
                extra_fields = {"group": group}
            elif self.data_args.template_id == 1:
                example[self.sentence2_key] = f"{example['question']} {example['answer']}"
        
        elif self.sentence1_key is not None and self.sentence2_key is not None:
            src_texts = [self.sentence1_key + ":", example[self.sentence1_key],
                         self.sentence2_key + ":", example[self.sentence2_key]]
        
        else:
            raise ValueError("The dataset preprocessing have value error.")
        
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields)

    def preprocessor_record_function(self, batch, add_prefix=True):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage
            passage = ex["passage"]
            passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            
            if add_prefix:
                inputs = self.name + " " + inputs
            
            # duplicates the samples based on number of answers.
            num_answers = len(ex["answer"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(ex["answers"] if num_answers > 0 else self.tokenizer.unk_token)
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend([{"answers": ex["answers"]}] * num_duplicates)
        
        return new_batch

    def preprocess_k_shot_dataset(self):
        None
    
    def split_dataset(self):
        # Training
        if self.training_args.do_train:
            self.train_dataset = self.tokenized_dataset["train"]
            if self.data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(self.data_args.max_train_samples))
        
        # Evaluation
        if self.training_args.do_eval:
            self.eval_dataset = self.tokenized_dataset["validation"]
            if self.data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.data_args.max_eval_samples))
        
        # Prediction
        if self.training_args.do_predict or self.name is not None or self.data_args.test_file is not None:
            self.predict_dataset = self.tokenized_dataset["test"]
            if self.data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(self.data_args.max_predict_samples))

    def set_metrics(self):
        if self.name is not None:
            self.metric = evaluate.load("super_glue", self.name)
        else:
            self.metric = evaluate.load("accuracy")
