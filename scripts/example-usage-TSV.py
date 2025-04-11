from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys
from typing import Callable, Mapping, List
import lemmatizer
import os
import json
from lemmatizer.TaggerLemmatizer import TaggerLemmatizer

# should be called tagger-lemmatizer-for-tokenized-tei
def main():
  t = TaggerLemmatizer(sys.argv[1])
  t.init()

  if len(sys.argv) > 3:
    t.tagTSV(sys.argv[2], sys.argv[3])
  else:
    t.tagPlainText('Dit is een zinnetje. En dit ook hoor!', '/tmp/out.txt')

if __name__ == '__main__':
    main()
