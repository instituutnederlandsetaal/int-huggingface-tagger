from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys
import lexicon_databases
from typing import Callable, Mapping, List
import lemmatizer
import os
import TaggerLemmatizer
import json

# should be called tagger-lemmatizer-for-tokenized-tei

if __name__ == '__main__':
  t = TaggerLemmatizer.TaggerLemmatizer("combi.config")
  t.init()

  if len(sys.argv) > 2:
    t.tagTEI(sys.argv[1], sys.argv[2])
  else:
    t.tagPlainText('Dit is een zinnetje. En dit ook hoor!', '/tmp/out.txt')

