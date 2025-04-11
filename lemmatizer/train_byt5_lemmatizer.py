

# zie https://gmihaila.github.io/tutorial_notebooks/pretrain_transformer/ ; https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py
# kijk ook eens naar https://github.com/RobinSmits/Dutch-NLP-Experiments
#### 0_training_args_byT5.py
from typing import Dict, List, Optional
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import HfArgumentParser,  TrainingArguments, set_seed, \
    AutoTokenizer, T5ForConditionalGeneration, Trainer,  MODEL_WITH_LM_HEAD_MAPPING
import re
import os
import sys

training_data = sys.argv[1] 
test_data = sys.argv[2]
output_dir = sys.argv[3]

dataset = load_dataset('csv', data_files={'train': [training_data], 'test': test_data}, sep="\t", download_mode='force_redownload')
print(dataset)


args_dict = {
    "model_name_or_path": 'google/byt5-small',
    "max_len": 128,
    "output_dir": output_dir, 
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "warmup_steps": 250,
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 250,
    "num_train_epochs": 5,
    "do_train": True,
    "do_eval": True,
    "fp16": False,
#    "use_cache": False,
#    "max_steps": 50000,
    "save_total_limit" : 2
}

# gekopieerd van de legacy

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )


    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )

    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_len : int = field(default=1000, metadata = {"help" : "maximum length"})


# einde kopie uit legacy language modeling

parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_dict(args_dict)
set_seed(training_args.seed)

#### 2_loading_model_and_tokenizer.py (naar voren verplaatst)

# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    max_length=data_args.max_len
)

model = T5ForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)

#### 1_dataset_loading_and_augmentation.py


print("DS42" + str(dataset["train"][42]))
dataset = dataset.map(lambda x: {'pos_word' : str(x['word']) + ' ' + str(x['pos']) }, batched=False)

train_dataset = dataset['train']
valid_dataset = dataset['test']

partly = False

if partly:
    print("select subsets to work with...")

    train_dataset = train_dataset.select(range(100000))
    valid_dataset = valid_dataset.select(range(10000))



# overwriting the default max_length of 20 
tokenizer.model_max_length=128
model.config.max_length=128

#### 3_reformat_dataset.py

print("start preparing datasets .....")

def prep_dataset(tokenizer, dataset, max_len):
    def convert_to_features(example_batch):

        x = example_batch['lemma']
        y = example_batch['pos_word']
    
        # print(f"x:{len(x)} y:{len(y)}")
        types = {}
        for i in range(0, len(x)):
          u = x[i]
          if u is None:
            print(f"None at {i}!!!!!")
            x[i] = '_'

        for z in x:
          types[type(z)] = 1

        # print(str(types))
        input_encodings = tokenizer.batch_encode_plus(example_batch['pos_word'],
                                                      truncation=True,
                                                      padding='max_length',
                                                      max_length=max_len
                                                      )
        target_encodings = tokenizer.batch_encode_plus(x, #example_batch['lemma'],
                                                       truncation=True,
                                                       padding='max_length',
                                                       max_length=max_len)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }

        return encodings

    dataset = dataset.map(convert_to_features, batched=True)
    # Set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'target_ids',
               'attention_mask', 'target_attention_mask']
    dataset.with_format(type='torch', columns=columns)
    # Rename columns to the names that the forward method of the selected
    # model expects
    dataset = dataset.rename_column('target_ids', 'labels')
    dataset = dataset.rename_column('target_attention_mask', 'decoder_attention_mask')
    # dataset = dataset.remove_columns(['text', 'ocr_text'])
    return dataset

train_dataset = prep_dataset(tokenizer, train_dataset, data_args.max_len)
valid_dataset = prep_dataset(tokenizer, valid_dataset, data_args.max_len)

print("datasets prepared, start training??")

#### 4_train_model.py

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train(
    model_path=model_args.model_name_or_path if os.path.isdir(
    model_args.model_name_or_path) else None
)

trainer.save_model()
# For convenience, we also re-save the tokenizer to the same directory,
# so that you can share your model easily on huggingface.co/models =)
tokenizer.save_pretrained(training_args.output_dir)
