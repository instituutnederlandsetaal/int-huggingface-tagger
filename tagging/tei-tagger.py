from transformers import AutoTokenizer, AutoModelForTokenClassification
from tokenizers import pre_tokenizers
from transformers import pipeline
import tei
import re
import sys
import os
from typing import Callable, Mapping, List
from functools import lru_cache
import nltk
import tei
import tagger

file = sys.argv[1]
output = sys.argv[2]
model_name = sys.argv[3]

tagger = tagger.find_model(model_name)
action  = lambda s : tagger.tag_tokenized_sentence(s)
tei.doWithTEISentences(file,output,action)
