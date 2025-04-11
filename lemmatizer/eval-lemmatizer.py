from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys
import lexicon_databases
from typing import Callable, Mapping, List
import lemmatizer
import os
import json
sys.path.append('../tagging')
import tei
import tagger
import eval_tagger

json_file =  open(sys.argv[1])
config = json.load(json_file)



lemmatizer = lemmatizer.Lemmatizer(config['lem_model'], config['lem_tokenizer'])
lemmatizer.init()



dataset_filename = sys.argv[2]
model_name = config['tagging_model']
model_filename_lastpart = re.sub('.*/(.)',r'\1', re.sub('/$','', model_name))
dataset_filename_lastpart= re.sub('.json', '', re.sub('.*/(.)',r'\1', re.sub('/$','', dataset_filename)))

eval_tagger.evaluate(model_name, dataset_filename, None, report = f'results/lemmaeval_{dataset_filename_lastpart}_{model_filename_lastpart}.log', lemmatizer=lemmatizer)

