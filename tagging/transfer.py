from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoModelForSequenceClassification, DataCollatorForLanguageModeling, AutoModelForMaskedLM, \
    AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
import evaluate
from torch import nn
import numpy as np
import sys
from datasets import load_metric, Dataset
from dataclasses import dataclass
import readFromTSV
from typing import Dict, List, Optional, Tuple
'''
Roughly follows https://huggingface.co/docs/transformers/tasks/token_classification

Fixing the bert layers: https://github.com/huggingface/transformers/issues/400
'''


def fix_lm_layers(model):
 for name, param in model.named_parameters():
   if 'classifier' not in name: # classifier layer
     param.requires_grad = False
   else:
     print(f"Not fixed: {name}", file=sys.stderr)

def tokenize_and_align_labels_x(tokenizer, examples, label_column_name, train_me_column_name):
    '''
    Cf https://huggingface.co/docs/transformers/tasks/token_classification:
    Adding the special tokens [CLS] and [SEP] and subword tokenization creates a mismatch between the input and labels.
    A single word corresponding to a single label may be split into two subwords.
    You will need to realign the tokens and labels by:
    - Mapping all tokens to their corresponding word with the word_ids method.
    - Assigning the label -100 to the special tokens [CLS] and [SEP] so the PyTorch loss function ignores them.
    - Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
    - 
    '''
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    #print(f"tokens={examples['tokens']}, tokenized_inputs={tokenized_inputs}")
    labels = []
    for i, (label,train_me) in enumerate(zip(examples[label_column_name],examples[train_me_column_name])):
        #print(f"Example:{i} label:{label} Compute loss:{train_me}")
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        #print(f"Word_ids: {word_ids}")
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None or train_me=="no" or train_me=="false": # Tell pytorch to ignore loss on these tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    #print(f"Labels={labels}")
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels(tokenizer, examples, label_column_name, train_me_column_name=None):
    '''
    Cf https://huggingface.co/docs/transformers/tasks/token_classification:
    Adding the special tokens [CLS] and [SEP] and subword tokenization creates a mismatch between the input and labels.
    A single word corresponding to a single label may be split into two subwords.

    You will need to realign the tokens and labels by:
    - Mapping all tokens to their corresponding word with the word_ids method.
    - Assigning the label -100 to the special tokens [CLS] and [SEP] so the PyTorch loss function ignores them.
    - Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
    '''

    # 'active learning' special case:
    if train_me_column_name is not None:
       return tokenize_and_align_labels_x(tokenizer, examples, label_column_name, train_me_column_name)

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    #print(f"tokens={examples['tokens']}, tokenized_inputs={tokenized_inputs}")
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        #print(f"Example:{i} label:{label}")
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        #print(f"Word_ids: {word_ids}")
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    #print(f"Labels={labels}")
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#redo this according to https://huggingface.co/docs/transformers/tasks/token_classification and run_ner.py; tokenizing does not work, use the tokenize_and_align_labels thing
#OK, the problem is that the format is per-sentence.
# pas op de ignore_mismatched_size vlag, kan dat kwaad voor het basisgeval??

def align_predictions(label_map, predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

@dataclass
class TaskTuning:
  model_name: str = None
  label_column_name: str = 'label'
  useModelNumLabels: bool = False 
  num_train_epochs: int =10
  train_me_column_name : str =None
  fix_bert_layers: bool =False
  # 
  label_list = None
  label_map = None
  per_epoch_evaluations = []

  def tasktune_token_classification_model(self,model_name, dataset: Dataset, label_column_name,output_dir,useModelNumLabels=False, num_train_epochs=10,train_me_column_name=None,fix_bert_layers=False,use_cpu=False):
      # print(f"labels:{label_column_name}, {dataset['train'].features[label_column_name]}")
      # label_list = dataset["train"].features[label_column_name].names # dit werkt dus niet meer om de een of andere reden
      label_list = self.label_list
      self.label_map = {i: label for i, label in enumerate(label_list)}
      num_labels = len(label_list)
      tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) # add prefix space only if roberta?
      #nl = model.config.num_labels if useModelNumLabels else num_labels
      model = AutoModelForTokenClassification.from_pretrained(model_name,num_labels=num_labels, ignore_mismatched_sizes=True) # ,num_labels=num_labels) # num_labels er bij geven??
  
      if fix_bert_layers:
         fix_lm_layers(model) 
  
      #print(f"{model_name} N Labels:{model.config.num_labels}, num_labels: {num_labels}, labels: {label_list}")
  
      tokenized_datasets: Dataset = dataset.map(lambda x: tokenize_and_align_labels(tokenizer,x, label_column_name,train_me_column_name=train_me_column_name), batched=True, remove_columns=[label_column_name, 'tokens'])
  
      small_train_dataset = tokenized_datasets["train"]#.shuffle(seed=42).select(range(1000))
      small_eval_dataset = tokenized_datasets["test"]#.shuffle(seed=42).select(range(1000))
  
      show_some_data = False
      if show_some_data:
          print(small_train_dataset[2])
          for f in small_train_dataset.features.keys():
              print(f'{f} --> {len(small_train_dataset[0][f])} :  {small_train_dataset[0][f]}')
  
  
      # evaluation metric
      useMetrics = True
      #if (useMetrics):
      #   accuracy = evaluate.load("accuracy")

      data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

      self.per_epoch_evaluations = []
      pev = self.per_epoch_evaluations

      def compute_metrics(p): # ToDo !important: deze klopt nog niet, voor nu weggelaten
          nonlocal pev
          preds_list, out_label_list = align_predictions(self.label_map, p.predictions, p.label_ids)
          # print(f"{preds_list} {out_label_list}")
          nTokens = 0
          nCorrect = 0
          for i in range(0, len(preds_list)):
             pred = preds_list[i]
             ref = out_label_list[i]
             for j in range(0, len(pred)):
                nTokens += 1
                if ref[j] == pred[j]:
                  nCorrect += 1
          a = nCorrect / nTokens 
          print(f"Accuracy {a} computed on {len(preds_list)} instances")
          pev.append(a)
          return {"accuracy" : a}
  
      # add label names to model config (otherwise you get LABEL_0 etc at inference time)
      model.config.label2id = {l: i for i, l in enumerate(label_list)}
      model.config.id2label = {i: l for i, l in enumerate(label_list)}
  
      #do the training
  
      training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch", save_total_limit=2, num_train_epochs=num_train_epochs, max_steps=-1, use_cpu=use_cpu)
  
      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=small_train_dataset,
          eval_dataset=small_eval_dataset,
          compute_metrics= compute_metrics if useMetrics else None,
          data_collator=data_collator
      )
      trainer.train()
  
      # save the model
      trainer.save_model()
      # an the tokenizer
      tokenizer.save_pretrained(training_args.output_dir)
      return self.per_epoch_evaluations

  def create_pos_dataset(self, training_data : str, test_data: str, pos_column: str='tags', data_format=None,fields=None, max_chunk_size=None) -> Dataset:
      """
      This assumes a file containing one json object per sentence, looking like this (data/tagging/tdn_train.json):
      {"id":"3",
      "tokens":["Nu","maar","weêr","verder",",","de","schaduwlooze","velden","weêr","in","."],
      "tags":["ADV(type=reg)","ADV(type=reg)","ADV(type=reg)","AA(degree=comp,position=free)","LET","PD(type=d-p,subtype=art,position=prenom)",
          "AA(degree=pos,position=prenom)","NOU-C(number=pl)","ADV(type=reg)","ADP(type=post)","LET"]}
      """
  
      trainit = [training_data] if isinstance(training_data,str) else training_data
    
      dataset = None
      if re.match(".*\.(tsv|tab|csv)(\.gz)?", test_data) or data_format=='tsv':
         print("Loading from TSV")
         dataset = readFromTSV.load_datasets_tsv(trainit, test_data, fields, max_chunk_size=max_chunk_size)
      else:
         dataset = load_dataset('json',
                             data_files={'train': trainit, 'test': test_data},
                             #sep="\t",
                             download_mode='force_redownload')
      z = get_label_list(dataset['train'][pos_column])  + get_label_list(dataset['test'][pos_column])


      label_list = list(set(z))
      label_list.sort()
      self.label_list = label_list

      label_to_id = {l: i for i, l in enumerate(label_list)}
  
      # print(str(label_to_id))
      (trainx, zzz) = self.create_classlabel(dataset['train'], pos_column, label_to_id)
      (testx, zzz) = self.create_classlabel(dataset['test'], pos_column, label_to_id)
  
      dataset['train'] = trainx
      dataset['test'] = testx
  
      print(dataset)
      return dataset

  def create_classlabel(self,dataset, label_column_name, label_to_id=None):
      '''
      Adds an integer 'label' column for the string labels in label_column_name
      :param dataset:
      :param label_column_name:
      :param label_to_id: predefined mapping from strings to ints
      :return: dataset with extra ClassLabel column in 'label'
      '''
      if label_to_id is None:
          label_list = get_label_list(dataset[label_column_name])
          print(str(dataset[label_column_name]) + " " + str(label_list))
          label_to_id = {l: i for i, l in enumerate(label_list)}
  
      label_list = list(label_to_id.keys())
      label_list.sort()
      self.label_list = label_list
      #print(label_list)
  
      num_labels = len(label_list)
      #print(label_to_id)
  
      dmapped = dataset.map(lambda row: {"label" : list(map(lambda x: label_to_id[x], row[label_column_name]))})
  
  
      #dmapped.features['label'].feature.names = label_list
      dmapped.features["label"] = ClassLabel(names=label_list)
  
      print_info = False
  
      if print_info:
          print(dmapped[1])
  
          for f in dmapped.features.keys():
              print(f'{f} --> {dmapped[1][f]}')
  
          print(dmapped.features)
  
      return (dmapped, label_to_id)

def finetune_language_model(model_name, dataset: Dataset, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True,is_split_into_words=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42) # .select(range(1000))
    small_train_dataset = small_train_dataset.filter(lambda x : len(x['text']) < 75 and not any(len(t) > 25 for t in x['text']))
    small_eval_dataset = small_eval_dataset.filter(lambda x:  len(x['text']) < 75 and not any(len(t) > 25 for t in x['text']))
    # prepare metrics.
    # TODO: these metrics do not work!!
    use_metrics = False
    compute_metrics = None

    if (use_metrics):
        metric = load_metric("accuracy")

        def compute_the_metrics(eval_preds): # van https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

        compute_metrics = compute_the_metrics

    # actually train

    training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch", save_total_limit=2)
  
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics= None, # compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    # save model

    trainer.save_model()

    # and tokenizer
    tokenizer.save_pretrained(training_args.output_dir)


# along the lines of https://github.com/xhan77/AdaptaBERT
# if, additionally. a smallish amount of labeled data is available for domain-specific, what is the best way to use it??

def domain_and_task_tuning(base_model_name, unlabeled_dataset, labeled_dataset, lm_output_dir, task_output_dir):
    # domain training
    finetune_language_model(base_model_name, unlabeled_dataset, lm_output_dir)
    tasktune_token_classification_model(lm_output_dir, labeled_dataset, task_output_dir)

def krantjes():
    finetune_krantjes()
    train_pos_dataset__krantjes()

# zie https://raw.githubusercontent.com/chriskhanhtran/spanish-bert/master/ner/run_ner.py, en ner_utils.py
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
# https://github.com/huggingface/transformers/blob/main/examples/legacy/token-classification/run_ner.py

from typing import List
def flatten(t: List[List[List[str]]]) -> List[List[str]]:
    return [item for sublist in t for item in sublist]

def get_label_list(labels):
    unique_labels = list(set(flatten(labels)))# list(set(labels))
    unique_labels.sort()
    return unique_labels

# dataset_with_duplicates = dataset.map(lambda batch: {"b": batch["a"] * 2}, remove_columns=["a"], batched=True)

def test_wnut():
    task_output_dir = '../data/tagging/wnut_ner_model'
    wnut = load_dataset("wnut_17")
    wnut['train'].to_json("/tmp/wnut.train.json.nw")
    wnut['test'].to_json("/tmp/wnut.test.json.nw")
    print(wnut)
    print(wnut['train'][100])
    tasktune_token_classification_model('bert-base-cased', wnut, label_column_name='ner_tags',
                                            output_dir=task_output_dir)


def train_pos_dataset():
    training_data = '../data/tagging/tdn_train.json'
    test_data = '../data/tagging/tdn_test.json'
    task_output_dir = '../data/tagging/pos_tagging_model_tdn'
    base_model = 'GroNLP/bert-base-dutch-cased'
    dataset = create_pos_dataset(training_data, test_data)
    tasktune_token_classification_model(base_model, dataset, label_column_name='label', output_dir=task_output_dir)

def train_pos_dataset__alpino():
    training_data = '../data/tagging/dutch-lassyklein-train.json'
    test_data = '../data/tagging/dutch-alpino-test.json'
    task_output_dir = '../data/tagging/pos_tagging_model_lassy'
    base_model = 'GroNLP/bert-base-dutch-cased'
    dataset = create_pos_dataset(training_data, test_data)
    tasktune_token_classification_model(base_model, dataset, label_column_name='label', output_dir=task_output_dir)

def train_pos_dataset__multatuli():
    training_data = '../data/tagging/dutch-lassyklein-train.json'
    test_data = '../data/tagging/dutch-alpino-test.json'
    task_output_dir = '../data/tagging/pos_tagging_model_multatuli'
    base_model = '../data/tagging/bertje2multatuli'
    dataset = create_pos_dataset(training_data, test_data)
    tasktune_token_classification_model(base_model, dataset, label_column_name='label', output_dir=task_output_dir)

def train_pos_dataset__krantjes():
    training_data = '../data/tagging/dutch-lassyklein-train.json'
    test_data = '../data/tagging/dutch-alpino-test.json'
    task_output_dir = '../data/tagging/pos_tagging_model_krantjes'
    base_model = '../data/tagging/bertje2kranten17'
    dataset = create_pos_dataset(training_data, test_data)
    tasktune_token_classification_model(base_model, dataset, label_column_name='label', output_dir=task_output_dir)

# multatuli_ideen.json

def finetune_multatuli():
    training_data = '../data/tagging/multatuli_ideen.json'
    test_data = '../data/tagging/kranten17.1644.json'
    dataset = load_dataset('json',
                           data_files={'train': [training_data], 'test': test_data},
                           #sep="\t",
                           download_mode='force_redownload')
    dataset = dataset.remove_columns('tags').rename_column('tokens', 'text')
    print(dataset)
    print(dataset['train'][1000])
    finetune_language_model('GroNLP/bert-base-dutch-cased', dataset,'../data/tagging/bertje2multatuli')

def finetune_krantjes():
    training_data = '../data/tagging/kranten17.1692.json'
    test_data = '../data/tagging/kranten17.1644.json'
    dataset = load_dataset('json',
                           data_files={'train': [training_data], 'test': test_data},
                           #sep="\t",
                           download_mode='force_redownload')
    dataset = dataset.remove_columns('tags').rename_column('tokens', 'text')
    print(dataset)
    print(dataset['train'][1000])
    finetune_language_model('GroNLP/bert-base-dutch-cased', dataset,'../data/tagging/language_models/bertje2kranten17')

def krantjes():
    finetune_krantjes()
    train_pos_dataset__krantjes()

def finetune_shakespeare():
    shakespeare = load_dataset("tiny_shakespeare")

    dataset = shakespeare.map(
        lambda batch: {'text': list(filter(lambda x: re.match('.*[a-z].*', x), batch['text'][0].split('\n')))},
        remove_columns=["text"],
        batched=True)
    print(dataset['train'][100])
    exit(0)
    output_dir = "test_finetune_shakespeare"
    model_name = "bert-base-cased"
    finetune_language_model(model_name, dataset, output_dir)

import re

if __name__=='__main__':
    krantjes()
    #finetune_multatuli()
    #train_pos_dataset__multatuli()
    #train_pos_dataset__alpino()
    #exit(0)

